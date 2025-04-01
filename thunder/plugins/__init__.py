from thunder.plugins.distributed import DDP, FSDP
from thunder.plugins.quantization import QuantizeInt4
from thunder.plugins.fp8 import FP8
from thunder.plugins.reduce_overhead import ReduceOverhead

names_to_plugins = {
    "ddp": DDP,
    "fsdp": FSDP,
    "quantize-int4": QuantizeInt4,
    "fp8": FP8,
    "reduce-overhead": ReduceOverhead,
}


def get_plugin(name):
    return names_to_plugins.get(name)


def get_plugin_names():
    return list(names_to_plugins.keys())


def register_plugin(name, cls):
    names_to_plugins[name] = cls
