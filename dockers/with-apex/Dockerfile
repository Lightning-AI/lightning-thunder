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

ARG APEX_CHECKOUT="master"

SHELL ["/bin/bash", "-c"]

RUN \
    # building Apex from source
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
    rm -rf apex

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import apex"
