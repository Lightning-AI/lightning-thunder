"""This is a modified bench.py from https://github.com/karpathy/nanogpt.

MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import time

import thunder
import torch
from thunder.tests.nanogpt_model import GPT, GPTConfig

# -----------------------------------------------------------------------------
config = "gpt2-medium"
batch_size = 16
seq_len = 128
bias = False
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = "float32"  #'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile_mode = "thunder"  # 'torch' or 'thunder'
print_loss = False  # print loss at every step
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

x = torch.randint(50304, (batch_size, seq_len), device=device)
y = torch.randint(50304, (batch_size, seq_len), device=device)
get_batch = lambda split: (x, y)

_configs = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
}

gptconf = GPTConfig(
    block_size=1024,  # how far back does the model look? i.e. context size
    vocab_size=50304,  # number of tokens
    dropout=0.1,
    **_configs[config],
)
model = GPT(gptconf)
model.to(device=device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95))

if compile_mode == "torch":
    print("Compiling model using torch.compile...")
    model = torch.compile(model)  # pytorch 2.0
elif compile_mode == "thunder":
    print("Compiling model using thunder.compile...")
    model = thunder.compile(model, use_rematerialization=True, use_static_caching=True, use_generated_backward=True)
else:
    raise ValueError(f"Unknown compile_mode: {compile_mode}")

# simple benchmarking
torch.cuda.synchronize()
for stage, num_steps in enumerate([10, 20]):  # burnin, then benchmark
    t0 = time.time()
    X, Y = get_batch("train")
    for k in range(num_steps):
        logits, loss = model(X, Y)
        X, Y = get_batch("train")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if print_loss:
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    if stage == 1:
        print(f"time per iteration: {dt/num_steps*1000:.4f}ms")
