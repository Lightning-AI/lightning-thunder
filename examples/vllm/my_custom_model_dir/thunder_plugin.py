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

# ---------------------------------------------------------------------
# 1.  low-level objects we want to intercept
# ---------------------------------------------------------------------
_VLLM_OVLD = torch.ops.vllm.unified_attention_with_output.default   # OpOverload
_VLLM_CAPS = torch.ops.vllm.unified_attention_with_output._op       # PyCapsule

# ---------------------------------------------------------------------
# 2.  Python wrapper – this is the *only* thing we expose to Thunder
#     (the helper below will generate the meta-function for us).
# ---------------------------------------------------------------------
def vllm_unified_attention(query, key, value, output, layer_name):
    """Thin eager wrapper that just unwraps proxies and calls the CUDA op"""
    q, k, v, o = map(unwrap, (query, key, value, output))
    _VLLM_OVLD(q, k, v, o, layer_name)          # writes into o in-place
    return output                               # hand the same proxy back

# ---------------------------------------------------------------------
# 3.  Build a *temporary* executor and publish it
# ---------------------------------------------------------------------
# --- after calling register_operator_for_opaque_function -------------
tmp_exec  = TemporaryExecutor()
thunder.extend.register_executor(tmp_exec)

# 1) create Symbol + meta + wrapper
_vllm_sym = tmp_exec.register_operator_for_opaque_function(
    vllm_unified_attention      # the eager wrapper we wrote earlier
)

# 2) make the tracer swap the C++ op for *that symbol*
#    (note: the value stored in _lookasides **is the symbol**, not the fn)
tmp_exec._lookasides[torch.ops.vllm.unified_attention_with_output.default] = _vllm_sym
tmp_exec._lookasides[torch.ops.vllm.unified_attention_with_output._op] = _vllm_sym

# ---------------------------------------------------------------------
# 4.  A tiny recipe that just adds our executor to the compile pass
# ---------------------------------------------------------------------
@thunder.Recipe.register("transformers_vllm")
class HFTransformersVLLM(BaseRecipe):
    compile_model = True

    def setup_executors(self):
        # keep whatever the base recipe already wants,
        # then append our temporary executor
        base = super().setup_executors() or []
        return base + [tmp_exec]

register_recipe("transformers_vllm", HFTransformersVLLM)

class ThunderForCausalLM(TransformersForCausalLM):
    _supports_attention_backend = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # compile the inner HF model *after* vLLM has patched it
        # (self.model is the TransformersModel instance)
        self._thunder_core = thunder.compile(
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

        hidden = self._thunder_core(           # compiled path
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden

    @classmethod
    def is_backend_compatible(cls):
        return True