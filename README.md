# Welcome to ⚡ Lightning Thunder

Lightning Thunder is a deep learning compiler for PyTorch. It makes PyTorch programs faster both on single accelerators or in distributed settings.

The main goal for Lightning Thunder is to allow optimizing user programs in the most extensible and expressive way possible.

**NOTE: Lightning Thunder is alpha and not ready for production runs.** Feel free to get involved, expect a few bumps along the way.

## What's in the box

Given a program, Thunder can generate an optimized program that:

- computes its forward and backward passes
- coalesces operations into efficient fusion regions
- dispatches computations to optimized kernels
- distributes computations optimally across machines

To do so, Lightning Thunder ships with:

- a JIT for acquiring Python programs targeting PyTorch and custom operations
- a multi-level IR to represent them as a trace of a reduced op-set
- an extensible set of transformations on the trace, such as `grad`, fusions, distributed (like `ddp`, `fsdp`), functional (like `vmap`, `vjp`, `jvp`)
- a way to dispatch operations to an extensible collection of executors

Lightning Thunder is written entirely in Python. Even its trace is represented as valid Python at all stages of transformation. This allows unprecedented levels of introspection and extensibility.

Lightning Thunder doesn't generate device code. It acquires and transforms user programs so that it's possible to optimally select or generate device code using fast executors like:

- [torch.compile](https://pytorch.org/get-started/pytorch-2.0/)
- [nvFuser](https://github.com/NVIDIA/Fuser)
- [cuDNN](https://developer.nvidia.com/cudnn)
- [Apex](https://github.com/NVIDIA/apex)
- [PyTorch eager](https://github.com/pytorch/pytorch) operations
- custom kernels, including those written with [OpenAI Triton](https://github.com/openai/triton)

Modules and functions compiled with Thunder fully interoperate with vanilla PyTorch and support PyTorch's autograd. Also, Thunder works alongside torch.compile to leverage its state-of-the-art optimizations.

## Install Thunder

Install the nvFuser nightly, which will also install the matching PyTorch nightly:

```bash
pip install --pre "nvfuser-cu121[torch]" --extra-index-url https://pypi.nvidia.com
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


cfoo = thunder.jit(foo)

a = torch.full((2, 2), 1)
b = torch.full((2, 2), 3)

result = cfoo(a, b)

print(result)

# prints
# tensor(
#  [[4, 4]
#   [4, 4]])
```

The compiled function `cfoo` takes and returns PyTorch tensors, just like the original function, so modules and functions compiled by Thunder can be used as part of larger PyTorch programs.

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
