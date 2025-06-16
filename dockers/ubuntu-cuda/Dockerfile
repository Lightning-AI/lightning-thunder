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

ARG UBUNTU_VERSION="24.04"
ARG CUDA_VERSION="12.6.3"
# select devel | runtime
ARG IMAGE_TYPE="devel"

FROM nvidia/cuda:${CUDA_VERSION}-${IMAGE_TYPE}-ubuntu${UBUNTU_VERSION}

ARG CUDNN_VERSION="9.8.0.87"
ARG CUDNN_FRONTEND_VERSION="1.10.0"
ARG PYTHON_VERSION="3.10"
ARG TORCH_VERSION="2.2.1"
ARG TRITON_VERSION="2.2.0"
ARG TORCH_INSTALL="stable"
ARG MAX_ALLOWED_NCCL=2.26.0

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER="GNU" \
    # compilation si too memory demanding so use only half of the cores
    MAKEFLAGS="-j$(( $(nproc) / 2 ))"

RUN \
    apt-get update -qq --fix-missing && \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    NCCL_VER=$(dpkg -s libnccl2 | grep '^Version:' | awk -F ' ' '{print $2}' | awk -F '-' '{print $1}' | grep -ve '^\s*$') && \
    echo "NCCL version found: $NCCL_VER" && \
    TO_INSTALL_NCCL=$(echo -e "$MAX_ALLOWED_NCCL\n$NCCL_VER" | sort -V | head -n1)-1+cuda${CUDA_VERSION_MM} && \
    echo "NCCL version to install: $TO_INSTALL_NCCL" && \
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
        liblapack-dev \
        openmpi-bin \
        graphviz \
        libnccl2=$TO_INSTALL_NCCL \
        libnccl-dev=$TO_INSTALL_NCCL \
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
    pip install "numpy >=1.23.0,<2" && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*;

# cuDNN package is used by both torch and cuDNN frontend python package.
# When using torch wheels, cuDNN pure-lib python package is fetched as a dependency.
# But when building torch from source, cuDNN needs to be manually installed first.
RUN if [ "${TORCH_INSTALL}" == "source" ]; then \
        echo "CUDA_VERSION=$CUDA_VERSION ; CUDNN_VERSION=$CUDNN_VERSION" && \
        CUDNN_BASE_VER=${CUDNN_VERSION%%.*} && \
        CUDA_VERSION_M=${CUDA_VERSION%%.*} && \
        apt update -qq --fix-missing && \
        CUDNN_PACKAGE_NAME="${CUDNN_VERSION}-1" && \
        apt upgrade -y --allow-downgrades --allow-change-held-packages && \
        apt install -y libcudnn9-cuda-${CUDA_VERSION_M}=${CUDNN_PACKAGE_NAME} \
                       libcudnn9-dev-cuda-${CUDA_VERSION_M}=${CUDNN_PACKAGE_NAME} \
                       nlohmann-json3-dev && \
        rm -rf /root/.cache && \
        rm -rf /var/lib/apt/lists/*; \
    fi

ENV \
    PYTHONPATH="/usr/lib/python${PYTHON_VERSION}/site-packages" \
    TORCH_CUDA_ARCH_LIST="8.0" \
    CUDA_SELECT_NVCC_ARCH_FLAGS="8.0"

ARG TORCH_INSTALL
ENV TORCH_USE_CUDA_DSA=1

# Build bitsandbytes 0.47.0-dev0 (0.46.0 has a bug)
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && \
        cd bitsandbytes && \
        git checkout 70bbbb9219cb10981c3b7f4457f30376e093d896 && \
        pip install pyproject.toml && \
        python setup.py install && \
        cd .. && \
        rm -rf bitsandbytes
RUN \
    if [ "${TORCH_INSTALL}" == "source" ]; then \
        # building pytorch from source
        git clone --recursive https://github.com/pytorch/pytorch && \
        cd pytorch && \
        git checkout "${TORCH_VERSION}" && \
        git submodule sync && \
        git submodule update --init --recursive && \
        pip install "cmake==3.31.6" -r requirements.txt && \
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
ENV NVFUSER_BUILD_NO_TEST=1
ENV NVFUSER_BUILD_NO_BENCHMARK=1

RUN \
    if [ "${TORCH_INSTALL}" == "source" ] || [ "${TORCH_INSTALL}" == "test" ]; then \
        # building nvFuser from source
        git clone --recursive https://github.com/NVIDIA/Fuser.git && \
        cd Fuser && \
        pip install -r requirements.txt && \
        pip install -v --no-build-isolation python/ && \
        cd .. && \
        rm -rf Fuser ; \
    else \
        # installing pytorch from wheels \
        CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
        TORCH_VERSION_MM=${TORCH_VERSION%.*} && \
        pip install --index-url https://pypi.nvidia.com/ -U "nvfuser-cu${CUDA_VERSION_MM/./}-torch${TORCH_VERSION_MM/./}" ; \
    fi

RUN \
    pip install nvidia-cudnn-frontend==${CUDNN_FRONTEND_VERSION}

RUN \
    # Show what we have \
    dpkg-query -W -f='${Package} ${Version}\n' libnccl2 libnccl-dev && \
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; print(f'PyTorch=={torch.__version__} with {torch.cuda.device_count()} GPUs')" && \
    python -c "import nvfuser; print(f'nvFuser=={nvfuser.version()}')" && \
    python -c "import triton; print(f'Triton=={triton.__version__}')"
