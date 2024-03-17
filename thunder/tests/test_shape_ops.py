import pytest
import torch
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.lang as tlang
import thunder.core.prims as prims
import thunder.langs.torch as ttorch

from .framework import executors, JAX_AVAILABLE, NOTHING, requiresJAX

if JAX_AVAILABLE:
    import jax

import numpy as np
