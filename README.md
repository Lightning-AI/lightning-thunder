# Welcome to ⚡ Lightning Thunder

`lightning.compile`, codenamed *thunder*, is a deep learning compiler for PyTorch. It makes PyTorch programs faster on single accelerators or in distributed settings.

*thunder* is intended to be fast, expressive, extensible, and easy to inspect.
It's written entirely in Python, and it runs PyTorch modules and functions using PyTorch plus fast executors like nvFuser, cuDNN Fusion Python API, Apex, and custom kernels (including those written with OpenAI Triton).

Modules and functions compiled with *thunder* also interoperate with vanilla PyTorch and support PyTorch's autograd.

*thunder* will be released publicly as a part of Lightning when it is in Beta.

## Build the documentation

Docs are currently not hosted publicly. However you can build them locally really quickly:

```bash
make docs
```

and point your browser to the generated docs at `docs/build/index.html`.

## Install *thunder*

If you qualify as an NVIDIA employee, hit the ground running through the pre-built container (docs at `docs/build/fundamentals/container.html`).

If not, follow the install instructions (docs at `docs/build/fundamentals/installation.html`).

## Hello World

Here is a simple example of how *thunder* lets you compile and run PyTorch modules and functions

```python
import torch
import thunder


def foo(a, b):
    return a + b


cfoo = thunder.compile(foo)

a = torch.full((2, 2), 1)
b = torch.full((2, 2), 3)

result = cfoo(a, b)

print(f"{result=}")

# prints
# result=tensor(
#  [[4, 4]
#   [4, 4]])
```

The compiled function `cfoo` takes and returns PyTorch tensors, just like the original function, so modules and functions compiled by *thunder* can be used as part of bigger PyTorch programs.

## FAQ

**Does *thunder* have any benchmarks?**

Yes! See `thunder/benchmarks` and the `litgpt-targets.py` and `nanogpt-targets.py` files for how to import and run them.

**How is *thunder* different from `torch.compile`?**

`torch.compile` is PyTorch’s native deep learning compiler. There are two principal differences
between `torch.compile` and *thunder*:

1. `thunder` is written entirely in Python
1. `thunder` has a multi-level IR that can reduce every operation to HLO-like primitives

These traits make `thunder` easy to debug, optimize and extend, enabling developers to quickly target optimized kernels or distributed strategies.

**How is thunder different from JAX?**

JAX is a deep learning compiler designed to work with XLA. There are two principal differences between JAX and thunder:

1. JAX exclusively traces functions, and does not have the equivalent of thunder's preprocessing to convert from PyTorch modules to traceable functions
1. *thunder* has a multi-level IR that makes it easy to reason about PyTorch operations, their decompositions and execution

[![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main?badge_token=mqheL1-cTn-280Vx4cJUdg)
