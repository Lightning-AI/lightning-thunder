import torch
import time
from thunder.transforms.quantization import BitsAndBytesLinearQuant4bit


def test_cpu_quantization():
    # Initialize quantization transform
    quant_transform = BitsAndBytesLinearQuant4bit()

    # Create a tensor on CPU
    weight = torch.randn(3, 3, device="cpu")

    # Quantize weight (expect only the quantized tensor, not a tuple)
    quantized_weight = quant_transform.quantize_weight(weight)

    # Check that the quantized tensor has fewer or equal elements due to compression
    original_num_elements = weight.numel()
    quantized_num_elements = quantized_weight.numel()

    assert quantized_weight is not None, "Quantized weight is None"
    assert (
        quantized_num_elements <= original_num_elements
    ), "Quantized tensor should have fewer or equal elements due to compression"


def test_gpu_quantization():
    if not torch.cuda.is_available():
        return

    # Initialize quantization transform
    quant_transform = BitsAndBytesLinearQuant4bit()

    # Create a tensor on GPU
    weight = torch.randn(3, 3, device="cuda")

    # Quantize weight (expect only the quantized tensor, not a tuple)
    quantized_weight = quant_transform.quantize_weight(weight)[0]

    # Check that the quantized tensor has fewer or equal elements due to compression
    original_num_elements = weight.numel()
    quantized_num_elements = quantized_weight.numel()

    assert quantized_weight is not None, "Quantized weight is None"
    assert (
        quantized_num_elements <= original_num_elements
    ), "Quantized tensor should have fewer or equal elements due to compression"


# Optional: Performance tests
def measure_time(device_type):
    quant_transform = BitsAndBytesLinearQuant4bit()

    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    weight = torch.randn(1000, 1000, device=device)

    start_time = time.time()
    quantized_weight = quant_transform.quantize_weight(weight)  # Expect only the quantized tensor
    end_time = time.time()

    print(f"Quantization time on {device_type}: {end_time - start_time:.4f} seconds")


# Run functional tests
print("Testing CPU quantization:")
test_cpu_quantization()

if torch.cuda.is_available():
    print("\nTesting GPU quantization:")
    test_gpu_quantization()
else:
    print("\nGPU not available, skipping GPU test.")

# Run performance tests
print("\nMeasuring performance:")
measure_time("cpu")
if torch.cuda.is_available():
    measure_time("cuda")
