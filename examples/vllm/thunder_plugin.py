
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
    def __init__(self):
        super().__init__(fuser="torch.compile")
        print("Applied recipe! 🧑‍🍳")

    def setup_executors(self):
        # keep whatever the base recipe already wants,
        # then append our temporary executor
        base = []
        return base + [tmp_exec]
        

register_recipe("transformers_vllm", HFTransformersVLLM)

import torch
import torch.nn.functional as F
import thunder

from typing import Optional, Union, List
# Make sure to import the necessary base class and types from vLLM
# This might vary slightly based on your vLLM version


class ThunderForCausalLM(TransformersForCausalLM):
    _supports_attention_backend = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.buckets: List[int] = [128, 256, 512, 1024, 2048, 4096, 8192]
        self.buckets = sorted(list(set(self.buckets)))
        
        self._compiled_buckets = torch.nn.ModuleDict()
        self._uncompiled_core = self.model
        
        # --- Pre-compilation at Startup ---
        self._precompile_all_buckets()

    def _precompile_all_buckets(self):
        """
        Creates dummy tensors for each bucket size and runs a forward pass
        to trigger and cache the JIT compilation for each one at startup.
        """
        print("--- Starting bucket pre-compilation ---")
        for bucket_size in self.buckets:
            bucket_key = f"prefill_{bucket_size}"
            print(f"Compiling PREFILL graph for bucket size: {bucket_size}...")
            
            # Create dummy inputs with the exact shape for this bucket
            dummy_input_ids = torch.zeros((1, bucket_size), dtype=torch.long)
            dummy_positions = torch.arange(bucket_size, dtype=torch.long).unsqueeze(0)
            
            # Compile the graph for this bucket
            self._compiled_buckets[bucket_key] = thunder.compile(
                self._uncompiled_core,
                recipe="transformers_vllm"
            )
            
            # Optional: You could do a warmup call here if needed, but compilation is the main goal.
            with torch.no_grad():
                self._compiled_buckets[bucket_key](input_ids=dummy_input_ids, positions=dummy_positions)

            print(f"Compilation for bucket {bucket_size} complete.")

        # Also compile the decode graph
        print("Compiling DECODE graph for sequence length 1...")
        self._compiled_buckets["decode_1"] = thunder.compile(
            self._uncompiled_core, recipe="transformers_vllm"
        )
        print("--- Pre-compilation finished ---")


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        # Get sequence length. We handle 1D and 2D inputs.
        seq_len = input_ids.shape[-1]
        
        # --- Decode Path (for single token generation) ---
        if seq_len == 1:
            compiled_fn = self._compiled_buckets["decode_1"]
            return compiled_fn(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        # --- Prefill Path (for prompt processing) ---
        is_1d_input = input_ids.ndim == 1
        if is_1d_input:
            input_ids = input_ids.unsqueeze(0)

        # Find the smallest bucket that fits the sequence length
        chosen_bucket = next((b for b in self.buckets if seq_len <= b), -1)
        if chosen_bucket == -1:
            raise ValueError(f"Seq len {seq_len} > largest bucket {self.buckets[-1]}")

        # Pad input_ids to the chosen bucket size
        padding_needed = chosen_bucket - seq_len
        padded_input_ids = F.pad(input_ids, (0, padding_needed), value=0)
        
        # --- THE CRITICAL FIX ---
        # Create a new, perfect positions tensor that exactly matches the
        # shape of our padded input_ids.
        batch_size = padded_input_ids.shape[0]
        padded_positions = torch.arange(chosen_bucket, dtype=torch.long, device=self.model.device)
        padded_positions = padded_positions.unsqueeze(0).expand(batch_size, -1)

        # Retrieve the correct pre-compiled graph
        bucket_key = f"prefill_{chosen_bucket}"
        compiled_fn = self._compiled_buckets[bucket_key]
        
        hidden_padded = compiled_fn(
            input_ids=padded_input_ids,
            positions=padded_positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=None,
        )

        # Slice the output back to the original sequence length
        hidden = hidden_padded[:, :seq_len, :]

        if is_1d_input:
            hidden = hidden.squeeze(0)

        return hidden