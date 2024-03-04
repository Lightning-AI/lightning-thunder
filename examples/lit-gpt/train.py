import time

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset

from thunder.tests.lit_gpt_model import GPT, Config

model_name = "open_llama_3b"
learning_rate = 6e-4
micro_batch_size = 2
max_iters = 50


def main(compile: str = "eager", dynamic: bool = False) -> None:
    fabric = L.Fabric(devices=1, precision="bf16-true")

    fabric.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP)

    config = Config.from_name(model_name)
    print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module():
        og_model = model = GPT(config)
    print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    if compile == "inductor":
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead", dynamic=dynamic)
    elif compile == "thunder":
        import thunder
        from thunder.executors.sdpaex import sdpa_ex
        from thunder.executors.torch_compile import torch_compile_executor

        model = thunder.jit(
            model,
            executors=[sdpa_ex, torch_compile_executor, thunder.nvfuser_executor, thunder.pytorch_executor],
            # TODO: we'd want to enable CUDAGraphs for parity with `torch.compile` but it goes OOM
        )
        model.max_seq_length = og_model.max_seq_length
    elif compile != "eager":
        raise ValueError(compile)

    model = fabric.setup(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-1, foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)

    train_data = DummyDataset(model.max_seq_length, dynamic)
    train_dataloader = DataLoader(train_data, batch_size=micro_batch_size, num_workers=2, collate_fn=pad_collate)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    train(fabric, model, optimizer, train_dataloader)
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(
    fabric: L.Fabric, model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader
) -> None:
    train_iter = iter(train_dataloader)
    t0 = None
    assert max_iters > 5
    for i in range(max_iters):
        iter_t0 = time.perf_counter()
        if i == 5:  # warmup
            t0 = iter_t0
        input_ids, targets = next(train_iter)

        logits = model(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        loss_item = loss.item()  # synchronization
        t1 = time.perf_counter()
        print(f"iter {i}: loss {loss_item:.4f}, iter time: {(t1 - iter_t0) * 1000:.2f}ms, t: {input_ids.size(1)}")
    print(f"Total time: {(t1 - t0):.2f}s")


class DummyDataset(IterableDataset):
    def __init__(self, max_seq_length: int, dynamic: bool):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.dynamic = dynamic

    def __iter__(self):
        while True:
            if self.dynamic:
                t = torch.randint(10, self.max_seq_length + 1, (1,))
            else:
                t = self.max_seq_length
            data = torch.randint(0, 100, (t + 1,), dtype=torch.int64)
            x = data[:t]
            y = data[1 : t + 1]
            yield x, y


def pad_collate(batch):
    x, y = zip(*batch)
    x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)
    return x_padded, y_padded


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
