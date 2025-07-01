
from typing import Optional, Union
import torch
import thunder
from vllm.model_executor.models.transformers import (
    TransformersForCausalLM, 
)
import torch, thunder
from thunder.extend          import TemporaryExecutor
from thunder.core.interpreter import unwrap


def vllm_unified_attention(query, key, value, output, layer_name):
    q, k, v, o = map(unwrap, (query, key, value, output))

    torch.ops.vllm.unified_attention_with_output.default(q, k, v, o, layer_name)

    return output, key, value # return tensors that were changed


def vllm_unified_attention_meta(query, key, value, output, layer_name):
    return output, key, value # return tensors that were changed

tmp_exec = TemporaryExecutor()
thunder.extend.register_executor(tmp_exec)

from thunder.core import prims

_vllm_sym = tmp_exec.register_operator(
    "vllm_unified_attention",
    meta=vllm_unified_attention_meta,
    fn=vllm_unified_attention,          
    tags=(prims.OpTags.DONT_DCE,) # needed to avoid DCE of unified_attention executor. TODO: don't rely on this.
)

tmp_exec._lookasides[torch.ops.vllm.unified_attention_with_output.default] = _vllm_sym
tmp_exec._lookasides[torch.ops.vllm.unified_attention_with_output._op] = _vllm_sym


import torch
import torch.nn.functional as F
import thunder

from typing import Optional, Union

class ThunderForCausalLM(TransformersForCausalLM):
    _supports_attention_backend = True # vllm requirement

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        # add needed executors
        from thunder.executors.nvfuserex import nvfuserex
        from thunder.executors.cudnnex import cudnn_ex
        
        self._core = thunder.jit(self.model, executors = [tmp_exec, nvfuserex, cudnn_ex], disable_torch_autograd=True)
        # if we don't disable the gradients here, the DONT_DCE optag leads to an error 

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[object] = None, # needed for model signature
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, object]:
        # this has two modes: one for seq_len > 1, and one for seq_len = 1.
        # This means it will compile twice on start.

        # padding
        # TODO: adapt for batch size > 1
        if input_ids.shape[0] > 1:

            current_seq_len = input_ids.shape[0]

            max_seq_len = self.vllm_config.scheduler_config.max_model_len

            pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

            padding_needed = max_seq_len - current_seq_len
            
            if padding_needed < 0:
                raise ValueError(f"Input sequence length ({current_seq_len}) exceeds max_seq_len ({max_seq_len})")

            input_to_core = F.pad(
                input_ids, (0, padding_needed), mode='constant', value=pad_token_id
            )

            positions_to_core = F.pad(
                positions, (0, padding_needed), mode='constant', value=0
            )

            out = self._core(input_to_core, positions_to_core)
            
            return out

        else:
            # --- DECODE LOGIC (Length == 1) ---
            return self._core(input_ids, positions)

            