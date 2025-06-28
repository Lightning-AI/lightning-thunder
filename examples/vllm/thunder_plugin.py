
from typing import Optional, Union
import torch
import thunder
from vllm.model_executor.models.transformers import (
    TransformersForCausalLM, IntermediateTensors,
)

import torch, thunder
from thunder.extend          import TemporaryExecutor
from thunder.core.interpreter import unwrap
from thunder.core.proxies     import proxy
from thunder.recipes          import BaseRecipe, register_recipe

# Your wrapper from before
def vllm_unified_attention(query, key, value, output, layer_name):
    """Thin eager wrapper that just unwraps proxies and calls the CUDA op"""
    q, k, v, o = map(unwrap, (query, key, value, output))
    # This op writes into o in-place
    torch.ops.vllm.unified_attention_with_output.default(q, k, v, o, layer_name)
    return output

# NEW: Let's write an explicit meta function for it
def vllm_unified_attention_meta(query, key, value, output, layer_name):
    """
    This is the contract with the compiler. It describes what the
    operator DOES, without actually running it.
    """
    # The unified_attention op returns its result by writing into the
    # `output` tensor. The function itself then returns this same tensor.
    # Therefore, the meta function should just return the `output`
    # tensor as-is, because its shape and dtype are the correct
    # representation of the function's output.
    # This also correctly models the side-effect on the `output` tensor.
    return output

# Now, let's update how we register it.
# We need to explicitly tell Thunder about our meta function.
# (The exact API might vary slightly with Thunder versions, but this is the principle)

tmp_exec = TemporaryExecutor()
thunder.extend.register_executor(tmp_exec)

_vllm_sym = tmp_exec.register_operator(
    "vllm_unified_attention",
    meta=vllm_unified_attention_meta  # Our NEW explicit meta function
)

# The rest of the lookaside registration remains the same
tmp_exec._lookasides[torch.ops.vllm.unified_attention_with_output.default] = _vllm_sym
tmp_exec._lookasides[torch.ops.vllm.unified_attention_with_output._op] = _vllm_sym

# Your recipe class remains the same

# ---------------------------------------------------------------------
# 4.  A tiny recipe that just adds our executor to the compile pass
# ---------------------------------------------------------------------
@thunder.Recipe.register("transformers_vllm")
class HFTransformersVLLM(BaseRecipe):
    compile_model = True

    def setup_executors(self):
        # keep whatever the base recipe already wants,
        # then append our temporary executor
        base = []
        return base + [tmp_exec]
        

register_recipe("transformers_vllm", HFTransformersVLLM)

class ThunderForCausalLM(TransformersForCausalLM):
    _supports_attention_backend = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # compile the inner HF model *after* vLLM has patched it
        # (self.model is the TransformersModel instance)
        self._thunder_core =  thunder.compile(
            self.model,                         # = self.transformer
            recipe="transformers_vllm",
        )

        # nothing else changes – forward() in the parent already
        # delegates to self.model, so it now hits the compiled graph.
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        print(input_ids.shape)
        hidden = self._thunder_core(           # compiled path
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        print(hidden)
        return hidden

    @classmethod
    def is_backend_compatible(cls):
        return True