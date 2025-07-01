import thunder_plugin
from vllm import ModelRegistry

# register the new model class
ModelRegistry.register_model("ThunderModel", thunder_plugin.ThunderForCausalLM)

def main():
    import torch, os
    import time

    # reproducibility params
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Save config with correct architectures locally
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    config = AutoConfig.from_pretrained("unsloth/Llama-3.2-1B")
    config.architectures = ["ThunderModel"]  # Match registry name
    config.auto_map = {
        "AutoModel": "thunder_plugin.ThunderForCausalLM",
        "AutoModelForCausalLM": "thunder_plugin.ThunderForCausalLM"
    }
    config.save_pretrained("my_custom_model_dir")
    AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B").save_pretrained("my_custom_model_dir")
    model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B", config=config)
    model.save_pretrained("my_custom_model_dir")

    # start vllm
    import torch, gc
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import destroy_model_parallel
    from vllm import LLM, SamplingParams

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        backend="nccl", distributed_init_method="file:///tmp/vllm_init"
    )
    initialize_model_parallel(1, 1)

    sp = SamplingParams(max_tokens=32, seed=42, temperature=0.0)
    llm = LLM(
        model="my_custom_model_dir",
        enforce_eager=True,
        max_model_len=256,
        max_num_batched_tokens=1024,
    )
    # warmup
    out_true = llm.generate(["Why is the sky blue?",], sp)
    print(out_true[0].outputs[0].text)

    start = time.time()
    out_true = llm.generate(["Why is the sky blue?",], sp)
    print(f"vllm with eager took {(time.time() - start):.4f}s")

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
        hf_overrides={"architectures": ["ThunderModel"]},
        enforce_eager=True,
        max_model_len=256,
        max_num_batched_tokens=256
    )

    # warmup
    out = llm.generate(["Why is the sky blue?",], sp)
    print(out[0].outputs[0].text)
    start = time.time()
    out = llm.generate(["Why is the sky blue?",], sp)
    print(f"vllm with thunder took {(time.time() - start):.4f}s")
    assert out[0].outputs[0].text == out_true[0].outputs[0].text

if __name__ == "__main__":
    main()
