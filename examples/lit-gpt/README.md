# Lit-GPT benchmarks

## Setup

```bash
wget -nc https://raw.githubusercontent.com/Lightning-AI/lit-gpt/main/scripts/download.py
wget -nc https://raw.githubusercontent.com/Lightning-AI/lit-gpt/main/scripts/convert_hf_checkpoint.py
wget -nc https://raw.githubusercontent.com/Lightning-AI/lit-gpt/main/scripts/prepare_openwebtext.py
pip install jsonargparse huggingface_hub sentencepiece tokenizers datasets
pip install git+https://github.com/Lightning-AI/lit-gpt
```

## [1 forward](1_forward.py)

```bash
python 1_forward.py --compile thunder
```

Runs a single forward call with a (B=10 x T=2048) tensor:

| Method    | Speed  | Memory  |
|-----------|--------|---------|
| Inductor  | 1.19 s | 17.3 GB |
| TinyLlama | 1.25 s | 18.8 GB |
| Thunder   | 1.30 s | 16.4 GB |
| Eager     | 1.48 s | 17.4 GB |

## [Compiled model inference](compiled_model_inference.py)

```shell
# setup
python download.py --repo_id openlm-research/open_llama_7b
python convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_7b
# run
python compiled_model_inference.py --compile thunder --fake false
```

Runs the existing generation logic with the model `forward` compiled:

| Method   | Speed        | Memory  |
|----------|--------------|---------|
| Inductor | 89.9 tok/sec | 13.8 GB |
| Eager    | 47.3 tok/sec | 13.6 GB |
| Thunder  | 45.5 tok/sec | 13.8 GB |

## [Compiled generation inference](compiled_generation_inference.py)

```shell
# setup
python download.py --repo_id openlm-research/open_llama_7b
python convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_7b
# run
python compiled_generation_inference.py --compile thunder --fake false
```

Runs a customized generation logic that is compiled and a customized multinomial implementation.
This is advantageous because `torch.multinomial(probs, num_samples=1)` is very slow. The model is also compiled:

| Method   | Speed        | Memory  |
|----------|--------------|---------|
| Inductor | 93.8 tok/sec | 13.8 GB |
| Eager    | 46.7 tok/sec | 13.6 GB |
| Thunder  | 45.5 tok/sec | 13.8 GB |

## [Training](train.py)

```shell
# setup
python download.py --repo_id openlm-research/open_llama_3b --tokenizer_only true
python prepare_openwebtext.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b --destination_path data/openwebtext-llama
# run
python train.py --compile thunder --dynamic false
```

Static shapes (45 iters)

| Method    | Speed  | Memory  |
|-----------|--------|---------|
| Inductor  | 20.3 s | 20.9 GB |
| Thunder   | 22.7 s | 28.0 GB |
| Eager     | 24.6 s | 24.2 GB |

Dynamic shapes (45 iters)

| Method    | Speed  |
|-----------|--------|
| Inductor  | 14.7 s |
| Eager     | 17.5 s |
| Thunder   | 1600 s |

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
torch==2.2.0.dev20231108+cu121
thunder==b108132a7e2c8d68791138db53c210357e8a9e76
nvfuser-cu121==0.1.2
```

TinyLlama:

```text
pytorch-triton==2.1.0+6e4932cda8
torch==2.1.0+cu121
flash-attn==2.2.2
xformers==0.0.22.post4
```

(Cannot upgrade `torch` because `flash-attn` breaks)
