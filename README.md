<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
    <br/>
<br/>

**Make PyTorch models Lightning fast.**

______________________________________________________________________

<p align="center">
  <a href="https://lightning.ai/">Lightning.ai</a> •
  <a href="#performance">Performance</a> •
  <a href="#get-started">Get started</a> •
  <a href="#install-thunder">Install</a> •
  <a href="#hello-world">Examples</a> •
  <a href="#features">Features</a> •
  <a href="#documentation">Documentation</a> •
</p>

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning-thunder/blob/main/LICENSE)
[![CI testing](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-thunder/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-thunder/badge/?version=latest)](https://lightning-thunder.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-thunder/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-thunder/main)

</div>

# Welcome to ⚡ Lightning Thunder

**Thunder makes PyTorch models Lightning fast.**

Thunder is a source-to-source compiler for PyTorch. It makes PyTorch programs faster by combining and using different hardware executors at once (ie: nvFuser, torch.compile, cuDNN, and TransformerEngine FP8).

Works on single accelerators and in multi-GPU settings.
Thunder aims to be usable, understandable, and extensible.

## Performance

Thunder can achieve significant speedups over standard PyTorch eager code, through the compounding effects of optimizations and the use of best-in-class executors. Here is an example of the pretraining throughput for Llama 2 7B as implemented in [LitGPT](https://github.com/Lightning-AI/litgpt).

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/training_throughput_single.png" width="800px" style="max-width: 100%;">
</div>

Thunder achieves a 40% speedup in training throughput compared to eager code on H100 using a combination of executors including nvFuser, torch.compile, cuDNN, and TransformerEngine FP8.

Thunder supports distributed strategies like DDP and FSDP (ZeRO2 and ZeRO3). Here is the normalized throughput measured for Llama 2 7B (this time without FP8 mixed precision, support for FSDP is underway).

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/normalized_training_throughput_zero2.png" width="800px" style="max-width: 100%;">
</div>

**NOTE: Lightning Thunder is alpha.** Feel free to get involved, expect a few bumps along the way.

## Get started

Try Thunder without installing by using our [Zero to Thunder Tutorial Studio](https://lightning.ai/lightning-ai/studios/zero-to-thunder-tutorial).

## Install Thunder

Install [nvFuser](https://github.com/NVIDIA/Fuser) nightly, and Thunder together

```bash
# install nvFuser which installs the matching nightly PyTorch
pip install --pre 'nvfuser-cu121[torch]' --extra-index-url https://pypi.nvidia.com

# install thunder
pip install lightning-thunder
```

<details>
  <summary>Advanced install options</summary>
    <!-- following section will be skipped from PyPI description -->

### Install from main

```bash
pip install git+https://github.com/Lightning-AI/lightning-thunder.git
```

### Install to tinker and contribute

Install this way to tinker with the internals and contribute:

```bash
pip install -e .
```

</details>
<!-- end skipping PyPI description -->

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

## Train models

Thunder is in its early stages and should not be used for production runs yet.

However, it can already deliver outstanding performance on LLM model supported by [LitGPT](https://github.com/Lightning-AI/lit-gpt), such as Mistral, Llama 2, Gemma, Falcon, and others.

Check out [the LitGPT integration](https://github.com/Lightning-AI/litgpt/tree/main/extensions/thunder) to learn about running LitGPT and Thunder together.

## Features

Given a Python callable or PyTorch module, Thunder can generate an optimized program that:

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
- Custom CUDA kernels through [PyCUDA](https://documen.tician.de/pycuda/tutorial.html#interoperability-with-other-libraries-using-the-cuda-array-interface), [Numba](https://numba.readthedocs.io/en/stable/cuda/kernels.html), [CuPy](https://docs.cupy.dev/en/stable/user_guide/kernel.html)
- Custom kernels written in [OpenAI Triton](https://github.com/openai/triton)

Modules and functions compiled with Thunder fully interoperate with vanilla PyTorch and support PyTorch's autograd. Also, Thunder works alongside torch.compile to leverage its state-of-the-art optimizations.

## Documentation

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
See the [LICENSE](LICENSE) file for details.
