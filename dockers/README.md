# Docker images

## Build images from Dockerfiles

You can build it on your own, note it takes lots of time, be prepared.

```bash
# build with specific arguments
docker image build -t lightning:base-cuda-py3.10-cuda12.1.1 -f dockers/base-cuda/Dockerfile --build-arg "CUDA_VERSION=12.1.1" .
```

To run your docker use

```bash
docker image list
docker run --rm -it pytorch-lightning:base-cuda-py3.10-cuda11.7.0 bash
```

## Run docker image with GPUs

To run docker image with access to your GPUs, you need to install

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

and later run the docker image with `--gpus=all`. For example,

```bash
docker run --rm -it --gpus=all pytorchlightning/lightning:base-cuda-py3.10-cuda12.1.0
```
