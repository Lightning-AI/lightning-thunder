# llama2.c

This is a copy of the Python bits in [llama2.c](https://github.com/karpathy/llama2.c) adapted to run with Thunder.
The repository uses the Llama 2 LLM architecture running on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
It's basically an evolution of [nanoGPT](https://github.com/karpathy/nanoGPT).

The scripts are configured to run with Thunder by default. You can `diff` them against their original versions to see the key changes.

## Setup

```shell
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.model
pip install -r requirements.txt
```

## Training

```shell
python tinystories.py download
python tinystories.py pretokenize
# 1 device
python train.py
# 2 devices (DDP)
torchrun --standalone --nproc_per_node=2 train.py
```

The code is configured to run with Thunder by default.

1 device:
* 319 ms/iter (thunder nvfuser)
* 343 ms/iter (inductor)
* 431 ms/iter (eager)

2 devices:
* 161 ms/iter (thunder nvfuser)
* 173 ms/iter (inductor)
* 217 ms/iter (eager)

CUDAGraphs are not used as the results were worse with them.

## Inference

```shell
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt -P out15M
python sample.py --checkpoint=out15M/stories15M.pt
```

## Setup

```shell
Thunder commit: 4aaa95858601ecac6faed74441d790158fbdeca4
Is debug build: False
CUDA used to build PyTorch: 12.1
CUDA runtime version: 12.1.105
GPU 0: NVIDIA A100-SXM4-40GB
Nvidia driver version: 525.105.17
pytorch-triton           2.1.0+6e4932cda8
torch                    2.1.0+cu121
nvfuser-cu121            0.0.19.dev20230925
```