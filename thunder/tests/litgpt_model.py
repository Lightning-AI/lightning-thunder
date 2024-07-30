"""Taken from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py"""

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


import litgpt

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
