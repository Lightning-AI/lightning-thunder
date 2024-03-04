# Lit-GPT benchmarks

## Setup

```bash
wget -nc https://raw.githubusercontent.com/Lightning-AI/lit-gpt/1a5e7c/scripts/download.py
pip install jsonargparse huggingface_hub sentencepiece tokenizers
pip install git+https://github.com/Lightning-AI/lit-gpt@1a5e7c
```

## [1 forward](1_forward.py)

```bash
python 1_forward.py --compile thunder
```

Runs a single forward call with a (B=10 x T=2048) tensor:

| Method   | Time ↓ | Memory ↓ |
|----------|--------|----------|
| Inductor | 1.18 s | 17.38 GB |
| Thunder  | 1.27 s | 16.32 GB |
| Eager    | 1.48 s | 17.44 GB |

## [Single-device training](train.py)

```shell
# setup
python download.py --repo_id openlm-research/open_llama_3b --tokenizer_only true
# run
python train.py --compile thunder --dynamic false
```

Static shapes (45 iters)

| Method   | Time ↓ | Memory ↓ |
|----------|--------|----------|
| Inductor | 20.1 s | 20.95 GB |
| Thunder  | 21.9 s | 23.75 GB |
| Eager    | 24.6 s | 24.28 GB |

Dynamic shapes (45 iters)

| Method   | Time ↓   | Memory ↓ |
|----------|----------|----------|
| Inductor | 17.0 s   | 20.69 GB |
| Eager    | 17.6 s   | 23.91 GB |
| Thunder  | ~5715 s  | -        |

## [Multi-device training](train_fsdp.py)

```shell
# setup
python download.py --repo_id openlm-research/open_llama_3b --tokenizer_only true
# run
python train_fsdp.py --devices 2 --compile thunder --stage 2 --bucketing_strategy BLOCK
```

Static shapes (45 iters)

| Stage | Bucketing | Method    | Time ↓  | Memory ↓ |
|-------|-----------|-----------|---------|----------|
| 2     | No        | Inductor  | Error   | Error    |
| 2     | No        | Thunder   | 23.29 s | 26.99 GB |
| 2     | No        | Eager     | 27.76 s | 27.61 GB |
|       |           |           |         |          |
| 2     | Block     | Inductor  | 21.71 s | 24.31 GB |
| 2     | Block     | Thunder   | 24.30 s | 26.96 GB |
| 2     | Block     | Eager     | 26.05 s | 27.67 GB |
|       |           |           |         |          |
| 3     | No        | Inductor  | Error   | Error    |
| 3     | No        | Thunder   | 24.39 s | 20.25 GB |
| 3     | No        | Eager     | 28.56 s | 20.75 GB |
|       |           |           |         |          |
| 3     | Block     | Inductor  | 21.76 s | 17.86 GB |
| 3     | Block     | Thunder   | 24.11 s | 26.93 GB |
| 3     | Block     | Eager     | 26.33 s | 21.23 GB |

## Setup

```text
Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Is debug build: False
CUDA used to build PyTorch: 12.1
CUDA runtime version: 12.3.107
GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 545.23.08

pytorch-triton==3.0.0+901819d2b6
torch==2.3.0.dev20240225+cu121
lightning-thunder==51993f9a6894f59f3779b30485e72b93d5e7b150
nvfuser_cu121==0.1.6.dev20240226
```
