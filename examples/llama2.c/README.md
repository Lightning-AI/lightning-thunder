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
git revert 4639efc03a0c7137a744abc8fe8b9bf9971a0e1d # 1293
# 1 device
python train.py
```

The code is configured to run with Thunder by default.

* ~342 ms/iter (inductor)
* ~350 ms/iter (thunder nvfuser)
* ~430 ms/iter (eager)

CUDAGraphs are not used as the results were worse with them.

## Inference

```shell
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt -P out15M
git revert 4639efc03a0c7137a744abc8fe8b9bf9971a0e1d # 1293
python sample.py --checkpoint=out15M/stories15M.pt
```

nanoGPT doesn't implement KV caching so this is expectedly slow. Please checkout the [Lit-GPT example](../lit-gpt/README.md) for faster text generation.

## Setup

```text
Thunder commit: 9761546938de49290c7d1472421fef844a89dfce
Is debug build: False
CUDA used to build PyTorch: 12.1
CUDA runtime version: 12.1.105
GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 525.105.17
pytorch-triton           2.1.0+6e4932cda8
torch                    2.2.0.dev20231108+cu121
nvfuser-cu121            0.1.2
```
