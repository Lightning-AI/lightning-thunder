![](docs/source/_static/images/lightning_thunder_lightmode_nobyline.png)

# Welcome to ⚡ Lightning Thunder

Lightning Thunder is a source-to-source compiler for PyTorch.

It makes PyTorch programs faster both on single accelerators or in distributed settings.

Thunder aims to be usable, understandable, and extensible.

## Performance

Thunder can achieve significant speedups over standard PyTorch eager code, through the compounding effects of optimizations and the use of best in class executors. Here is an example of the pretraining throughput for Llama 2 7B as implemented in [LitGPT](https://github.com/Lightning-AI/litgpt).

![](docs/source/_static/images/training_throughput_single.png)

We achieve a 40% speedup in training throughput compared to eager code on H100 using a combination of executors including nvFuser, torch.compile, cuDNN, and TransformerEngine FP8.

Thunder supports distributed strategies like DDP and FSDP (ZeRO2 and ZeRO3). Here is the normalized throughput measured for Llama 2 7B (this time without FP8 mixed precision, support for FSDP is underway).

![](docs/source/_static/images/normalized_training_throughput_zero2.png)

**NOTE: Lightning Thunder is alpha.** Feel free to get involved, expect a few bumps along the way.

## Start with Thunder

Try Thunder without installing by using our [Zero to Thunder Tutorial Studio](https://lightning.ai/lightning-ai/studios/zero-to-thunder-tutorial).

## Install Thunder

Install [nvFuser](https://github.com/NVIDIA/Fuser) nightly, which will also install the matching PyTorch nightly:

```bash
pip install --pre 'nvfuser-cu121[torch]' --extra-index-url https://pypi.nvidia.com
```

Install Thunder:

```bash
pip install git+https://github.com/Lightning-AI/lightning-thunder.git
```

or install from the local repo:

```bash
pip install .
```

## Hello World

Here is a simple example of how Thunder lets you compile and run PyTorch code:

```python
import torch
import thunder


def foo(a, b):
    return a + b


jfoo = thunder.jit(foo)

a = torch.full((2, 2), 1)
b = torch.full((2, 2), 3)

result = jfoo(a, b)

print(result)

# prints
# tensor(
#  [[4, 4]
#   [4, 4]])
```

The compiled function `jfoo` takes and returns PyTorch tensors, just like the original function, so modules and functions compiled by Thunder can be used as part of larger PyTorch programs.

## Running training

Thunder is in its early stages, it should not be used for production runs yet.

However, it can already deliver outstanding performance on models supported by [LitGPT](https://github.com/Lightning-AI/lit-gpt), such as Mistral, Llama2, Gemma, Falcon, and derivatives.

Run training loop for Llama, single-GPU:

```bash
python examples/lit-gpt/train.py
```

Run training loop for Llama, multi-GPU, using FSDP:

```bash
python examples/lit-gpt/train_fsdp.py
```

See [README.md](examples/lit-gpt/README.md) for details on running LitGPT with Thunder.

## What's in the box

Given a python callable or PyTorch module, Thunder can generate an optimized program that:

- Computes its forward and backward passes
- Coalesces operations into efficient fusion regions
- Dispatches computations to optimized kernels
- Distributes computations optimally across machines

To do so, Thunder ships with:

- A JIT for acquiring Python programs targeting PyTorch and custom operations
- A multi-level IR to represent operations as a trace of a reduced op-set
- An extensible set of transformations on the trace, such as `grad`, fusions, distributed (like `ddp`, `fsdp`), functional (like `vmap`, `vjp`, `jvp`)
- A way to dispatch operations to an extensible collection of executors

Thunder is written entirely in Python. Even its trace is represented as valid Python at all stages of transformation. This allows unprecedented levels of introspection and extensibility.

Thunder doesn't generate code for accelerators directly. It acquires and transforms user programs so that it's possible to optimally select or generate device code using fast executors like:

- [torch.compile](https://pytorch.org/get-started/pytorch-2.0/)
- [nvFuser](https://github.com/NVIDIA/Fuser)
- [cuDNN](https://developer.nvidia.com/cudnn)
- [Apex](https://github.com/NVIDIA/apex)
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
- [PyTorch eager](https://github.com/pytorch/pytorch)
- custom kernels, including those written with [OpenAI Triton](https://github.com/openai/triton)

Modules and functions compiled with Thunder fully interoperate with vanilla PyTorch and support PyTorch's autograd. Also, Thunder works alongside torch.compile to leverage its state-of-the-art optimizations.

## Build the documentation

Docs are currently not hosted publicly. However you can build them locally really quickly:

```bash
make docs
```

and point your browser to the generated docs at `docs/build/index.html`.

## Develop and run tests

You can set up your environment for developing Thunder by installing the development requirements:

```bash
pip install -r requirements/devel.txt
```

Install Thunder as an editable package (optional):

```bash
pip install -e .
```

Now you run tests:

```bash
pytest thunder/tests
```

Thunder is very thoroughly tested, so expect this to take a while.

## License

Lightning Thunder is released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
See LICENSE file for details.

[![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main?badge_token=mqheL1-cTn-280Vx4cJUdg)
