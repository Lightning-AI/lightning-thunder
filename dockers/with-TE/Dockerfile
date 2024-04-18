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

ARG BASE_IMAGE_TAG

FROM pytorchlightning/lightning-thunder:${BASE_IMAGE_TAG}

ARG TRANSFORMER_ENGINE_CHECKOUT="stable"

SHELL ["/bin/bash", "-c"]

# compilation si too memory demanding so use only half of the cores
ENV MAKEFLAGS="-j$(( $(nproc) / 3 ))"

RUN \
    # building TE from source
    git clone --branch ${TRANSFORMER_ENGINE_CHECKOUT} \
      --recursive https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    git submodule update --init --recursive && \
    export NVTE_FRAMEWORK=pytorch && \
    pip install . && \
    cd .. && \
    rm -rf TransformerEngine

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import transformer_engine"
