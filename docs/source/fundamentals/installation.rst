Install Lightning Thunder
#########################

Minimal dependencies
====================

Follow these instructions to install PyTorch, nvFuser, and finally Thunder.

Install PyTorch and nvFuser with pip (command shown is for CUDA 12.1)::

  pip install --pre "nvfuser-cu121[torch]" --extra-index-url https://pypi.nvidia.com

cu121 can be replaced with cu118 depending on your CUDA version.

You're all set with minimal dependencies, so you can follow `Install Thunder`_.

Dependencies so far don't include additional optimized kernels for execution like OpenAI Triton, cuDNN Fusion, or Apex.
These are described in the following section, `Optional dependencies`_.

Optional dependencies
=====================

Install Apex
------------

Thunder can use NVIDIA's Apex to accelerate some PyTorch operations. To install the Apex executor, first clone the Apex git repository and then execute the following command in the project's root directory::

  git clone https://github.com/NVIDIA/apex.git
  cd apex

  pip install -v --no-cache-dir --no-build-isolation --config-settings "--build-option=--xentropy" ./

Install cuDNN
-------------

Thunder can use NVIDIA's cuDNN Python frontend bindings to accelerate some PyTorch operations. cuDNN backend is a pure-lib python package which needs to downloaded separately.

  pip install nvidia-cudnn-cu12
  pip install nvidia-cudnn-frontend

You're all set, now follow `Install Thunder`_.

Install OpenAI Triton
---------------------

Thunder can easily integrate OpenAI Triton kernels. You can install Triton using::

  pip install triton


Install Thunder
===============

You can now install Thunder::

  pip install git+https://github.com/Lightning-AI/lightning-thunder.git

Alternatively you can clone the Thunder repository and install locally::

  git clone https://github.com/Lightning-AI/lightning-thunder.git
  cd lightning-thunder

  pip install .
