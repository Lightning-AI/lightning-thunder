#!/bin/bash

pip list
python -c "import torch ; assert torch.cuda.is_available(), 'missing GPU support!'"
python -c "import torch ; v = torch.__version__ ; assert str(v).startswith('2'), v"
python -c "from thunder.executors.utils import nvfuser_available ; assert nvfuser_available(), 'nvFuser is missing!'"
python -c "from thunder.executors.triton_utils import triton_version ; assert triton_version() is not None, 'triton is missing!'"
