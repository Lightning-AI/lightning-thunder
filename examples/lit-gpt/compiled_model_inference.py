import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config

from boring_bits import Tokenizer
from thunder.tests.lit_gpt_model import GPT


@torch.inference_mode()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    T = idx.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


@torch.inference_mode()
def main(
    checkpoint_dir: Path = Path("checkpoints/openlm-research/open_llama_7b"),
    num_samples: int = 10,
    max_new_tokens: int = 200,
    compile: str = "eager",
    fake: bool = True,
) -> None:
    torch.set_float32_matmul_precision("high")

    fabric = L.Fabric(devices=1, precision="bf16-true")

    tokenizer = Tokenizer(Path(checkpoint_dir))
    encoded = tokenizer.encode("Hello, my name is", device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    with fabric.init_module(empty_init=False):
        og_model = model = GPT.from_name(checkpoint_dir.name)
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # initialize the kv cache for the first time
        model.set_kv_cache(batch_size=1)
    if not fake:
        checkpoint = torch.load(checkpoint_dir / "lit_model.pth")
        model.load_state_dict(checkpoint)
    model.eval()

    if compile == "inductor":
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    elif compile == "thunder":
        import thunder
        from thunder.executors.utils import Executor

        executors = [Executor.TORCH]
        executors = [Executor.NVFUSER] + executors
        model = thunder.compile(
            og_model, disable_torch_autograd_support=True, use_cudagraphs=True, executors_list=executors
        )
        model.max_seq_length = og_model.max_seq_length
    elif compile != "eager":
        raise ValueError(compile)

    model = fabric.setup_module(model)

    values = []
    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        if not fake:
            fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        tok_per_sec = tokens_generated / t
        values.append(tok_per_sec)
        fabric.print(f"Time for inference {i + 1}: {t:.02f} sec total, {tok_per_sec:.02f} tokens/sec")
        # reset the kv cache
        for block in og_model.transformer.h:
            block.attn.kv_cache.k.zero_()
            block.attn.kv_cache.v.zero_()
    print(f"Best: {max(values):05f}")
    fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
