# Lit-GPT benchmarks

## Setup

```bash
wget -nc https://raw.githubusercontent.com/Lightning-AI/lit-gpt/main/scripts/download.py
wget -nc https://raw.githubusercontent.com/Lightning-AI/lit-gpt/main/scripts/convert_hf_checkpoint.py
pip install jsonargparse huggingface_hub sentencepiece tokenizers
pip install git+https://github.com/Lightning-AI/lit-gpt
```

## [1 forward](1_forward.py)

```bash
python 1_forward.py --compile thunder
```

Runs a single forward call with a (B=10 x T=2048) tensor:

| Method   | Time ↓ | Memory ↓ |
|----------|--------|----------|
| Inductor | 1.18 s | 17.3 GB  |
| Thunder  | 1.26 s | 16.3 GB  |
| Eager    | 1.48 s | 17.4 GB  |

## [Compiled model inference](compiled_model_inference.py)

```shell
# setup
python download.py --repo_id meta-llama/Llama-2-7b-hf
python convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
# run
python compiled_model_inference.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --compile thunder --fake false
```

Runs the existing generation logic with the model `forward` compiled:

| Method   | Speed ↑      | Memory ↓ |
|----------|--------------|----------|
| Inductor | 89.5 tok/sec | 13.8 GB  |
| Eager    | 46.7 tok/sec | 13.6 GB  |
| Thunder  | 40.0 tok/sec | 13.8 GB  |

## [Compiled generation inference](compiled_generation_inference.py)

```shell
# setup
python download.py --repo_id meta-llama/Llama-2-7b-hf
python convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
# run
python compiled_generation_inference.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --compile thunder --fake false
```

Runs a customized generation logic that is compiled and a customized multinomial implementation.
This is advantageous because `torch.multinomial(probs, num_samples=1)` is very slow. The model is also compiled:

| Method   | Speed ↑      | Memory ↓ |
|----------|--------------|----------|
| Inductor | 93.5 tok/sec | 13.8 GB  |
| Eager    | 46.5 tok/sec | 13.6 GB  |
| Thunder  | 40.0 tok/sec | 13.8 GB  |

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
| Inductor | 20.3 s | 20.9 GB  |
| Thunder  | 22.1 s | 23.6 GB  |
| Eager    | 24.6 s | 24.2 GB  |

Dynamic shapes (45 iters)

| Method   | Time ↓ |
|----------|--------|
| Inductor | 14.7 s |
| Eager    | 17.5 s |
| Thunder  | 1600 s |

## [Multi-device training](train_fsdp.py)

```shell
# setup
python download.py --repo_id openlm-research/open_llama_3b --tokenizer_only true
# run
python train_fsdp.py --devices 2 --compile thunder --stage 2
```

Static shapes (45 iters)

| Stage | Method    | Time ↓ | Memory ↓ |
|-------|-----------|--------|----------|
| 2     | Inductor  | 23.2 s | 24.3 GB  |
| 2     | Thunder   | 23.5 s | 27.0 GB  |
| 2     | Eager     | 27.7 s | 27.6 GB  |
|       |           |        |          |
| 3     | Inductor  | 24.8 s | 17.5 GB  |
| 3     | Thunder   | 25.8 s | 20.2 GB  |
| 3     | Eager     | 28.5 s | 20.7 GB  |

## Setup

```text
Python version: 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] (64-bit runtime)
Is debug build: False
CUDA used to build PyTorch: 12.1
CUDA runtime version: 12.1.105
GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 525.125.06

pytorch-triton==2.3.0.dev20240115+cu121
torch==2.3.0.dev20240115+cu121
lightning-thunder==7adf3e56e00ad52f9214437828512bc3de89277e
nvfuser_cu121==0.1.5.dev20240116
```
