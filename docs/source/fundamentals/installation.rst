Install Thunder
###############

Minimal thunder installation
============================

Follow these instructions to install PyTorch, torchvision, nvFuser, and thunder. This installation will only include PyTorch and nvFuser for execution and does not include any other library which could provide further optimized kernels for execution with OpenAI Triton, cuDNN Fusion, or APex. See the following sections for information on how to install those with a Thunder source build.

Install dependencies
--------------------

Install PyTorch, torchvision, and nvFuser with pip (command shown is for CUDA 12.1)::

  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121;
  pip install nvfuser-cu121;
  patch-nvfuser

You may need to specify a different CUDA version.

Install thunder
---------------

*thunder* is not yet public, and it must be installed from source.

Request access to the *thunder* repository (see the :doc:`NVIDIA Point of Contact section <get_involved>` for how to do so), then clone the *thunder* repository::

  git clone https://github.com/Lightning-AI/lightning-thunder.git
  cd lightning-thunder

execute the following command in the project root directory::

  python setup.py develop

Full thunder installation with all executors
============================================

thunder is a Python package that supports Python 3.10.x and depends on PyTorch 2.1.x+. Thunder can be built with only PyTorch, however it is best used with other executors that run some operations faster than PyTorch. This section describes how to install thunder and all of its current executors.

Install PyTorch
---------------

thunder depends on PyTorch 2.1.x+, which can be installed from source or using pip. See the PyTorch GitHub for instructions on how to install from source, or download a nightly wheel following the instructions at pytorch.org.

Install nvFuser
---------------

thunder relies on nvFuser to fuse CUDA operations. To install nvFuser, first install PyTorch (see above) then run the following commands (for CUDA 12.1)::

  pip install nvfuser-cu121
  patch-nvfuser

cu121 can be replaced with cu118 or cu117 depending on your CUDA version.

Install Apex
------------

*thunder* can use NVIDIA's Apex to accelerate some PyTorch operations. To install the Apex executor, first clone the Apex git repository and then execute the following command in the project's root directory::

  pip install -v --no-cache-dir --no-build-isolation --config-settings "--build-option=--xentropy" ./

Install cuDNN
-------------

*thunder* can use NVIDIA's cuDNN Python frontend bindings to accelerate some PyTorch operations. cuDNN's Python frontend currently requires being built from source. See the Git repository for instructions, below is a template for CUDA 12.x that requires setting the ``CUDAToolkit_ROOT`` environment variable::

  pip install nvidia-cudnn-cu12
  export CUDNN_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/
  for file in $CUDNN_PATH/lib/*.so.[0-9]; do filename_without_version="${file%??}"; ln -s $file $filename_without_version; done

  git clone -b 1.0/pre_release_2 https://github.com/NVIDIA/cudnn-frontend.git
  export CUDAToolkit_ROOT=/path/to/cuda
  CMAKE_BUILD_PARALLEL_LEVEL=16 pip install cudnn_frontend/ -v

Install thunder
---------------

*thunder* is not yet public, and it must be installed from source.

Request access to the *thunder* repository (see the :doc:`NVIDIA Point of Contact section <get_involved>` for how to do so), then clone the *thunder* repository::

  git clone https://github.com/Lightning-AI/lightning-thunder.git
  cd lightning-thunder

execute the following command in the project root directory::

  python setup.py develop
