import thunder_plugin
from vllm import ModelRegistry

ModelRegistry.register_model("ThunderModel", thunder_plugin.ThunderForCausalLM)

def main():
    import thunder_plugin
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    # Save config with correct architectures field
    config = AutoConfig.from_pretrained("unsloth/Llama-3.2-1B")
    config.architectures = ["ThunderModel"]  # Match registry name
    config.auto_map = {
        "AutoModel": "thunder_plugin.ThunderForCausalLM",
        "AutoModelForCausalLM": "thunder_plugin.ThunderForCausalLM"
    }
    config.save_pretrained("my_custom_model_dir")
    from vllm import ModelRegistry
    ModelRegistry.register_model("ThunderModel", thunder_plugin.ThunderForCausalLM)  # Match config
    

    AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B").save_pretrained("my_custom_model_dir")

    model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B", config=config)
    model.save_pretrained("my_custom_model_dir")

    from vllm import LLM

    import torch, thunder, gc
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import destroy_model_parallel
    from vllm import LLM, SamplingParams

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        backend="nccl", distributed_init_method="file:///tmp/vllm_init"
    )
    initialize_model_parallel(1, 1)

    llm = LLM(
        model="my_custom_model_dir",
        hf_overrides={"architectures": ["ThunderModel"]},
        enforce_eager=True
    )

    sp = SamplingParams(max_tokens=128, temperature=0.0)
    out = llm.generate(["Why is the sky blue?",], sp)
    print(out[0].outputs[0].text)

    import gc

    import torch
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel

    # Delete the llm object and free the memory
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory!")

    llm = LLM(
        model="my_custom_model_dir",
        enforce_eager=True
    )
    out = llm.generate(["Why is the sky blue?",], sp)
    print(out[0].outputs[0].text)

    torch.distributed.destroy_process_group()
if __name__ == "__main__":
    main()
