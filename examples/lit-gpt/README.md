# Lit-GPT benchmarks

## [1 forward](1_forward.py)

Runs a single forward call with a (B=10 x T=2048) tensor:

| Method    | Speed  | Memory  |
|-----------|--------|---------|
| Inductor  | 1.19 s | 17.3 GB |
| TinyLlama | 1.25 s | 18.8 GB |
| Thunder   | 1.29 s | 16.4 GB |
| Eager     | 1.48 s | 17.4 GB |

## [Compiled model inference](compiled_model_inference.py)

Runs the existing generation logic with the model `forward` compiled:

| Method   | Speed        | Memory  |
|----------|--------------|---------|
| Inductor | 89.9 tok/sec | 13.8 GB |
| Thunder  | 74.2 tok/sec | 13.8 GB |
| Eager    | 47.3 tok/sec | 13.6 GB |

FIXME: thunder generation is not correct

You will need to download the tokenizer data for the checkpoint. You can use https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/download.py for this.

## [Compiled generation inference](compiled_generation_inference.py)

Runs a customized generation logic that is compiled and a customized multinomial implementation.
This is advantageous because `torch.multinomial(probs, num_samples=1)` is very slow. The model is also compiled:

| Method    | Speed        | Memory      |
|-----------|--------------|-------------|
| Inductor  | 93.8 tok/sec | 13.8 GB     |
| Thunder   | Unsupported  | Unsupported |
| Eager     | 46.7 tok/sec | 13.6 GB     |

## [Training](train.py)

Static shapes

| Method    | Speed       | Memory      |
|-----------|-------------|-------------|
| Inductor  | 452 ms/iter | 20.9 GB     |
| TinyLlama | 492 ms/iter | 27.8 GB     |
| Thunder   | Unsupported | Unsupported |
| Eager     | 548 ms/iter | 24.2 GB     |

Dynamic shapes (45 iters)

| Method    | Speed       |
|-----------|-------------|
| Inductor  | 14.7 s      |
| Thunder   | Unsupported |
| Eager     | 17.5 s      |

## Setup

```shell
Is debug build: False
CUDA used to build PyTorch: 12.1
CUDA runtime version: 12.1.105
GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 525.125.06
```

Inductor and Thunder

```text
pytorch-triton==2.1.0+6e4932cda8
torch==2.2.0.dev20231102+cu121
thunder==d56afda7c9ad18477991487e2fbd5398942525d8
nvfuser-cu121==0.1.1.dev20231030
```

TinyLlama:

```text
pytorch-triton==2.1.0+6e4932cda8
torch==2.1.0+cu121
flash-attn==2.2.2
xformers==0.0.22.post4
```

(Cannot upgrade `torch` because `flash-attn` breaks)