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

- ~215 ms/iter (torch.compile 'inductor')
- ~239 ms/iter (thunder nvfuser)
- ~339 ms/iter (eager)

CUDAGraphs are not used as the results were worse with them.

## Inference

```shell
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt -P out15M
python sample.py --checkpoint=out15M/stories15M.pt
```

nanoGPT doesn't implement KV caching so this is expectedly slow. Please checkout the [Lit-GPT example](../lit-gpt/README.md) for faster text generation.

## Setup

```text
Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Is debug build: False
CUDA used to build PyTorch: 12.4
CUDA runtime version: 12.4.99
GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 550.54.14

triton == 3.0.0
torch == 2.4.0a0+git685ace3
nvfuser @ 0.2.0+git70101da
```
