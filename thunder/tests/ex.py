import torch
import thunder

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available with {torch.cuda.device_count()} GPU(s)")
    device = 'cuda'
else:
    print("CUDA is not available, using CPU")
    device = 'cpu'

# Example 1: Using 3D tensors (batch dimension + matrix dimensions)
# Shape: [batch_size, rows, cols]
A = torch.randn(3, 32, device=device, dtype=torch.bfloat16)  # 4 matrices of size 3x5
B = torch.randn(32, 64, device=device, dtype=torch.bfloat16)  # 4 matrices of size 5x6
C = torch.tensor([0, 0], device=device, dtype=torch.int32)

print(f"3D tensor example:")
print(f"Input A shape: {A.shape}")
print(f"Input B shape: {B.shape}")
print(f"Tensors are on device: {A.device}")

# Call torch._grouped_mm with 3D tensors
result_3d = torch._grouped_mm(A, B, C)
print(f"3D result shape: {result_3d.shape}")  # Expected: [4, 3, 6]
print("torch._grouped_mm with 3D tensors executed successfully")
# Alternative using bmm for batch matrix multiplication
expected_3d = torch.bmm(A, B)
print(f"Alternative 3D result shape: {expected_3d.shape}")
# torch.testing.assert_close(result_3d, expected_3d, rtol=1e-5, atol=1e-6)

print("\n")


def test_2d_mm(a, b):
    return torch._grouped_mm(a, b)

result_3d = test_2d_mm(A, B)
print(f"Result shape from test_2d_mm: {result_3d.shape}")
another_result = thunder.jit(test_2d_mm)(A, B)
print("Result shape from thunder.jit: %s\n", another_result.shape)

