# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VERSION="22.04"
ARG CUDA_VERSION="12.1.0"
# select devel | runtime
ARG IMAGE_TYPE="devel"

FROM nvidia/cuda:${CUDA_VERSION}-${IMAGE_TYPE}-ubuntu${UBUNTU_VERSION}

ARG CUDNN_VERSION="8.9.7.29-1"
ARG CUDNN_FRONTEND_CHECKOUT="v1.3.0"
ARG PYTHON_VERSION="3.10"
ARG TORCH_VERSION="2.2.1"
ARG TRITON_VERSION="2.2.0"
ARG TORCH_INSTALL="stable"
ARG APEX_CHECKOUT="master"
ARG TRANSFORMER_ENGINE_CHECKOUT="stable"

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER="GNU" \
    MAKEFLAGS="-j$(nproc)"

RUN \
    apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        build-essential \
        ca-certificates \
        software-properties-common \
        pkg-config \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
        unzip \
        libopenmpi-dev \
        openmpi-bin \
        graphviz \
        libnccl2 \
        libnccl-dev \
        ssh \
    && \
    # Install python
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/* \

RUN \
    echo "CUDA_VERSION=$CUDA_VERSION ; CUDNN_VERSION=$CUDNN_VERSION " && \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    # There are some test failures from cuDNN 12.1, so 'upgrade' requests for 12.1 to 12.2.
    CUDA_VERSION_MM="${CUDA_VERSION_MM/12.1/12.2}" && \
    CUDNN_BASE_VER=${CUDNN_VERSION%%.*} && \
    CUDNN_PACKAGE_VER="${CUDNN_VERSION}+cuda${CUDA_VERSION_MM}" && \
    apt update -qq --fix-missing && \
    apt upgrade -y --allow-change-held-packages \
      libcudnn${CUDNN_BASE_VER}=${CUDNN_PACKAGE_VER} \
      libcudnn${CUDNN_BASE_VER}-dev=${CUDNN_PACKAGE_VER} \
    && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/* && \
    git clone --branch ${CUDNN_FRONTEND_CHECKOUT} https://github.com/NVIDIA/cudnn-frontend.git && \
    CMAKE_BUILD_PARALLEL_LEVEL=16 pip install cudnn-frontend/ -v && \
    rm -rf cudnn-frontend

COPY requirements/ requirements/

ENV \
    PYTHONPATH="/usr/lib/python${PYTHON_VERSION}/site-packages" \
    TORCH_CUDA_ARCH_LIST="8.0" \
    CUDA_SELECT_NVCC_ARCH_FLAGS="8.0"

ARG TORCH_INSTALL
ENV TORCH_USE_CUDA_DSA=1

RUN \
    if [ "${TORCH_INSTALL}" == "source" ]; then \
        # building pytorch from source
        git clone --recursive https://github.com/pytorch/pytorch && \
        cd pytorch && \
        git checkout "${TORCH_VERSION}" && \
        git submodule sync && \
        git submodule update --init --recursive && \
        pip install . && \
        pip install "pytorch-triton==$(cat .ci/docker/triton_version.txt)" --index-url="https://download.pytorch.org/whl/nightly/" && \
        cd .. && \
        rm -rf pytorch; \
    elif [ "${TORCH_INSTALL}" == "test" ]; then \
        # installing pytorch from wheels
        CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
        pip install "torch==${TORCH_VERSION}" "triton==${TRITON_VERSION}" \
          --index-url="https://download.pytorch.org/whl/test/cu${CUDA_VERSION_MM//'.'/''}"; \
    else \
        # installing pytorch from wheels \
        CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
        pip install "torch==${TORCH_VERSION}" "triton==${TRITON_VERSION}" \
          --index-url="https://download.pytorch.org/whl/cu${CUDA_VERSION_MM//'.'/''}"; \
    fi

ARG TORCH_INSTALL

RUN \
    if [ "${TORCH_INSTALL}" == "source" ]; then \
        # building nvFuser from source
        git clone https://github.com/NVIDIA/Fuser.git && \
        cd Fuser && \
        git submodule update --init --recursive && \
        pip install -r requirements.txt && \
        python setup.py install --no-test --no-benchmark && \
        cd .. && \
        rm -rf Fuser ; \
    elif [ "${TORCH_INSTALL}" == "test" ]; then \
        # building nvFuser from source
        git clone https://github.com/NVIDIA/Fuser.git && \
        cd Fuser && \
        git submodule update --init --recursive && \
        pip install -r requirements.txt && \
        python setup.py install --no-test --no-benchmark && \
        cd .. && \
        rm -rf Fuser ; \
    else \
        # installing pytorch from wheels \
        CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
        TORCH_VERSION_MM=${TORCH_VERSION%.*} && \
        pip install -U "nvfuser-cu${CUDA_VERSION_MM/./}-torch${TORCH_VERSION_MM/./}" ; \
    fi

RUN \
    # building Apex from source \
    if [ -n "$TRANSFORMER_ENGINE_CHECKOUT" ]; then \
        git clone --branch ${TRANSFORMER_ENGINE_CHECKOUT} \
          --recursive https://github.com/NVIDIA/TransformerEngine.git && \
        cd TransformerEngine && \
        git submodule update --init --recursive && \
        export NVTE_FRAMEWORK=pytorch && \
        pip install . && \
        cd .. && \
        rm -rf TransformerEngine \
    fi \

RUN \
    # building Apex from source \
    if [ -n "$APEX_CHECKOUT" ]; then \
        pip install "pip>=23.1" packaging && \
        git clone https://github.com/NVIDIA/apex && \
        cd apex && \
        git checkout ${APEX_CHECKOUT} && \
        # https://github.com/NVIDIA/apex#linux
        pip install -v \
          --disable-pip-version-check \
          --no-cache-dir \
          --no-build-isolation \
          --config-settings "--build-option=--xentropy" \
          . && \
        cd .. && \
        rm -rf apex \
    fi

RUN \
    ls -lh requirements/ && \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    pip install -U -r requirements/test.txt && \
    rm -rf requirements/

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; print(f'PyTorch=={torch.__version__} with {torch.cuda.device_count()} GPUs')" && \
    python -c "import nvfuser; print(f'nvFuser=={nvfuser.version()}')" && \
    python -c "import triton; print(f'Triton=={triton.__version__}')"
