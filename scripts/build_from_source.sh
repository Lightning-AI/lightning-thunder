# Usage:
#   $ bash build_from_source.sh <optional flags>

set -euo pipefail

CUDA_VERSION_DEFAULT=11.7

VALID_ARGS=$(getopt -o h -l help,cuda:,use_kineto,use_distributed -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -h | --help)
        echo "bash build_from_source.sh"
        echo "  --cuda             The version of CUDA to install and build with. (default ${CUDA_VERSION_DEFAULT})"
        echo "  --use_kineto       Build PyTorch with Kineto support"
        echo "  --use_distributed  Build PyTorch with distributed support"
        exit 0
        ;;
    --cuda)
        CUDA_VERSION=$2 ; shift 2 ;;
    --use_kineto)
        USE_KINETO=1 ; shift ;;
    --use_distributed)
        USE_DISTRIBUTED=1 ; shift ;;
    --) shift;
        break
        ;;
  esac
done

set -x

SCRIPT_FOLDER="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
BUILD="${SCRIPT_FOLDER}/build"
mkdir -p ${BUILD}

# Helpers to raide better error messages.
function _error_msg {
  printf '%s\n' "FAIL: $1" >&2
  : "${__fail_fast:?$1}";
}

function check_cmd {
  $(command -v $1 &> /dev/null) || _error_msg "'$1' not present"
}

# Check the conda environment.
check_cmd "conda"
[ -v CONDA_PREFIX ] || _error_msg "Not in a conda environment."
[ "${CONDA_DEFAULT_ENV:-base}" != "base" ] || _error_msg "Do not mess with the base environment."

# Install latest version of git so submodule commands work reliably
conda install -y git
conda update git

# =============================================================================
# == nvFuser ==================================================================
# =============================================================================
NVFUSER_PYTORCH="${BUILD}/pytorch_for_nvfuser"
if [ ! -d "${NVFUSER_PYTORCH}" ]; then
  git clone https://github.com/csarofeen/pytorch.git "${NVFUSER_PYTORCH}"
fi

pushd "${NVFUSER_PYTORCH}"
git checkout devel;
git fetch origin devel;
git reset --hard origin/devel;
popd

export NVFUSER_SOURCE_DIR="${NVFUSER_PYTORCH}/third_party/nvfuser"

# =============================================================================
# == CUDA / NVCC (Try to respect existing install) ============================
# =============================================================================
CUDA_VERSION=${CUDA_VERSION:-$CUDA_VERSION_DEFAULT}
if [ ! $(command -v nvcc) ]; then
  printf "\n\nNote: This might take 5+ minutes.\n\n"
  time conda install -y -c conda-forge cudatoolkit-dev=${CUDA_VERSION}
fi

# Compile a trivial kernel to ensure CUDA has everything it needs.
HELLO_WORLD_CU="${BUILD}/hello_world.cu"
cat << EOF > ${HELLO_WORLD_CU}
__global__ void kernel (void){}

int main(void){
  kernel <<<1,1>>>();
  return 0;
}
EOF
check_cmd "nvcc"
nvcc ${HELLO_WORLD_CU} -o "${BUILD}/hello_world.a" || \
_error_msg "Failed to compile trivial CUDA program. Check your NVCC."

# =============================================================================
# == PyTorch ==================================================================
# =============================================================================
PYTORCH="${BUILD}/pytorch"
if [ ! -d "${PYTORCH}" ]; then
  git clone https://github.com/pytorch/pytorch.git "${PYTORCH}"
fi

# Purge any previous torch installs. Sometimes you need multiple pip uninstalls.
conda uninstall pytorch -y || true
pip uninstall --yes torch;
pip uninstall --yes torch;

pushd "${PYTORCH}"
git submodule sync;
git checkout master;
git fetch origin; 
git reset --hard origin/master;
make clean || true
python setup.py clean; 
git clean -xfdf;
git submodule foreach --recursive git reset --hard; 
git submodule update --init --recursive
git submodule foreach --recursive git clean -xfdf;

conda install -y cmake ninja
conda update cmake
pip install -r "requirements.txt"
time \
  USE_DISTRIBUTED=${USE_DISTRIBUTED:-0} \
  BUILD_CAFFE2=0 \
  BUILD_CAFFE2_OPS=0 \
  BUILD_TEST=0 \
  USE_FBGEMM=0 \
  USE_NNPACK=0 \
  USE_XNNPACK=0 \
  USE_QNNPACK=0 \
  USE_PYTORCH_QNNPACK=0 \
  USE_KINETO=${USE_KINETO:-0} \
  REL_WITH_DEB_INFO=1 \
  python setup.py develop --cmake
popd

# =============================================================================
# == Thunder ==================================================================
# =============================================================================
pushd "${SCRIPT_FOLDER}/.."
pip install -r "requirements.txt"
conda install -y pytest
python setup.py develop
popd

# Validate
python "${SCRIPT_FOLDER}/validate_build.py"
