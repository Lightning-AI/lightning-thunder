import time

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config

from thunder.tests.lit_gpt_model import GPT


@torch.inference_mode()
def main(name: str = "open_llama_7b", num_samples: int = 10, compile: str = "eager") -> None:
    torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda")

    with device:
        model = GPT.from_name(name)
        # tinyllama support
        if not hasattr(model, "max_seq_length"):
            model.max_seq_length = model.config.block_size
        encoded = torch.randint(0, model.config.padded_vocab_size, (10, model.max_seq_length))

    model.eval()

    if compile == "inductor":
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        model = torch.compile(model, fullgraph=True)  # , mode="reduce-overhead")
    elif compile == "thunder":
        import thunder

        executors = [thunder.pytorch_executor]
        executors = [thunder.nvfuser_executor] + executors
        model = thunder.compile(
            model, disable_torch_autograd_support=True, use_cudagraphs=False, executors_list=executors
        )
    elif compile != "eager":
        raise ValueError(compile)

    values = []
    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        _ = model(encoded)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        values.append(t)
        print(f"Time for inference {i + 1}: {t:.05f} sec total")
    print(f"Best: {min(values):05f}")
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
