import transformers
import thunder
import time
from litgpt import Config, GPT
import torch
import torch.utils.benchmark
from quantization_transform import QuantizedLinearTransform, nvfp4_executor
from torchao.quantization import quantize_
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4InferenceConfig,
    NVFP4MMConfig,
)


model_name = "Llama-3-8B"
device = "cuda"
N_LAYER = 1

cfg: Config = Config.from_name(model_name)
# cfg.n_layer = N_LAYER


def benchmark_model(model, inp, name):
    torch.cuda.reset_peak_memory_stats()
    allocated_memory = torch.cuda.memory_allocated()

    timer = torch.utils.benchmark.Timer(
        stmt="model(inp)",
        setup="",
        globals={"model": model, "inp": inp},
        label=f"Llama-3-8B Input Shape {inp.shape}",
        description=name,
    )

    measurement = timer.timeit(number=10)
    print(f"{name} Time taken: {measurement}")
    print(f"Allocated memory: {allocated_memory / 1e9} GB")
    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")
    print(f"Extra memory allocated: {(torch.cuda.max_memory_allocated() - allocated_memory) / 1e9} GB")
    print()
    return measurement

for BS in [1, 2, 4, 8, 16, 32]:
    inp = torch.randint(0, 255, (BS, 2048,), device=device)

    # Initialize in the loop as the model is updated inplace later in the loop.
    with torch.device(device):
        model = GPT(cfg).to(torch.bfloat16)
    model.eval().requires_grad_(False)

    eager_measurement = benchmark_model(model, inp, "Eager")

    compiled_model = thunder.jit(model)

    thunder_default_measurement = benchmark_model(compiled_model, inp, "Thunder default")

    trcs = thunder.last_traces(compiled_model)
    trcs[-1].save_trace("thunder_default_trc.py")

    cmodel = torch.compile(model)

    torch_compile_measurement = benchmark_model(cmodel, inp, "TorchCompile Default")

    def quantization_filter(name, module):
        return isinstance(module, torch.nn.Linear) # and "mlp" in name
        # return "mlp" in name and isinstance(module, torch.nn.Linear)

    xforms = [QuantizedLinearTransform(filter_fn=quantization_filter, separate_quantization=True)]
    executors = (nvfp4_executor,) + thunder.get_default_executors()

    compiled_model = thunder.jit(model, transforms=xforms, executors=executors)

    thunder_measurement = benchmark_model(compiled_model, inp, "Thunder + nvFP4")

    trcs = thunder.last_traces(compiled_model)
    trcs[-1].save_trace("quantized_trc.py")

    mm_config = NVFP4MMConfig.DYNAMIC
    config = NVFP4InferenceConfig(
            mm_config=mm_config, use_triton_kernel=True
    )

    # This mutates the model
    quantize_(model, config=config)

    torchao_measurement = benchmark_model(model, inp, "Torchao")

    cmodel = torch.compile(model)

    torchao_compile_measurement = benchmark_model(cmodel, inp, "TorchCompile + AO")

    compare = torch.utils.benchmark.Compare([eager_measurement, thunder_default_measurement, thunder_measurement,
                                            torch_compile_measurement, torchao_measurement, torchao_compile_measurement])
    compare.print()
