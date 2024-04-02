"""Taken from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py"""

import torch
import torch.nn as nn


configs = [
    # diverse sample of configs FOR TESTING that cover all major checkpoints variants architecturally but with reduced
    # size
    dict(name="gpt-neox-like", block_size=128, n_layer=2, n_embd=64, n_head=4, padding_multiple=8),
    dict(
        name="llama1-like",
        block_size=128,
        vocab_size=320,
        padding_multiple=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        mlp_class_name="LLaMAMLP",
        intermediate_size=1376,
    ),
    dict(
        name="long-context-like",
        block_size=512,
        vocab_size=320,
        padding_multiple=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=4,
    ),
    dict(
        name="llama2-like",
        vocab_size=320,
        padding_multiple=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=1376,
    ),
    dict(
        name="falcon-7b-like",
        block_size=128,
        padded_vocab_size=254,
        n_layer=2,
        n_head=7,
        n_embd=448,
        rotary_percentage=1.0,
        n_query_groups=1,
        bias=False,
        shared_attention_norm=True,
    ),
    dict(
        name="falcon-40b-like",
        block_size=128,
        padded_vocab_size=508,
        n_layer=2,
        n_head=64,
        n_embd=256,
        rotary_percentage=1.0,
        n_query_groups=4,
        bias=False,
    ),
    dict(
        name="codellama2-like",
        block_size=1024,
        vocab_size=2001,
        padding_multiple=16,
        n_layer=2,
        n_head=4,
        n_embd=64,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMLP",
        intermediate_size=1376,
        rope_base=1000000,
    ),
    dict(
        name="mixtral-like",
        block_size=512,
        padded_vocab_size=500,
        n_layer=2,
        n_head=64,
        n_embd=256,
        rotary_percentage=1.0,
        n_query_groups=8,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-05,
        mlp_class_name="LLaMAMoE",
        intermediate_size=224,
        rope_base=1000000,
        n_expert=8,
        n_expert_per_token=2,
    ),
]

name_to_config = {config["name"]: config for config in configs}


class OverridenKVCache(nn.Module):
    def __init__(
        self,
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        if torch._dynamo.is_compiling():
            # inductor doesn't support `index_add` with bfloat16
            k = self.k.index_copy_(2, input_pos, k)
            v = self.v.index_copy_(2, input_pos, v)
            return k, v
        # See issue: "Support more indexing operators (index_copy and index_add)"
        k = self.k = torch.index_add(self.k, 2, input_pos, k)
        v = self.v = torch.index_add(self.v, 2, input_pos, v)
        # THUNDER bug: cannot return self.k, self.v here (may be cuda graphs related - no minimum repro)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


import litgpt

# override for operator workarounds
litgpt.model.KVCache = OverridenKVCache
# add the testing configurations
litgpt.config.name_to_config.update(name_to_config)
name_to_config.update(litgpt.config.name_to_config)

# manually expose for backwards compatibility
Config = litgpt.Config
GPT = litgpt.GPT
RMSNorm = litgpt.model.RMSNorm
CausalSelfAttention = litgpt.model.CausalSelfAttention
LLaMAMLP = litgpt.model.LLaMAMLP
build_rope_cache = litgpt.model.build_rope_cache
apply_rope = litgpt.model.apply_rope
Block = litgpt.model.Block
