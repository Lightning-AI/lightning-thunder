# llama2.c

This is a copy of the Python bits in [llama2.c](https://github.com/karpathy/llama2.c) adapted to run with Thunder.
The repository uses the Llama 2 LLM architecture running on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
It's basically an evolution of [nanoGPT](https://github.com/karpathy/nanoGPT).

The scripts are configured to run with Thunder by default. You can `diff` them against their original versions to see the key changes.

## Setup

```shell
wget -nc https://github.com/karpathy/llama2.c/raw/master/tokenizer.model
pip install -r requirements.txt
```

## Training

```shell
python tinystories.py download
python tinystories.py pretokenize
# 1 device
python train.py --compile=thunder  # thunder|eager|torch
# multiple devices, 1 node
torchrun --standalone --nproc_per_node=2 train.py --compile=thunder
```

The code is configured to run with Thunder by default.

Results with 1 GPU:

- ~339 ms/iter (torch.compile 'inductor')
- ~347 ms/iter (thunder nvfuser)
- ~431 ms/iter (eager)

CUDAGraphs are not used as the results were worse with them.

## Inference

```shell
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt -P out15M
python sample.py --checkpoint=out15M/stories15M.pt
```

nanoGPT doesn't implement KV caching so this is expectedly slow. Please checkout the [Lit-GPT example](../lit-gpt/README.md) for faster text generation.

## Setup

```text
Python version: 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] (64-bit runtime)
Is debug build: False
CUDA used to build PyTorch: 12.1
CUDA runtime version: 12.1.105
GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 525.125.06

pytorch-triton @ https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2B901819d2b6-cp310-cp310-linux_x86_64.whl
torch @ https://download.pytorch.org/whl/nightly/cu121/torch-2.3.0.dev20240130%2Bcu121-cp310-cp310-linux_x86_64.whl
lightning-thunder==8b107c6fe531c94c6705dbf39700863685ba5b65
nvfuser_cu121==0.1.5.dev20240131
```
