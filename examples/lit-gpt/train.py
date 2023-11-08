import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from thunder.tests.lit_gpt_model import GPT, Config

model_name = "open_llama_3b"
name = "openwebtext-llama"
out_dir = Path("out") / name
data_dir = Path("data") / name
learning_rate = 6e-4
micro_batch_size = 2
max_iters = 50


def main(compile: str = "eager", dynamic: bool = False) -> None:
    fabric = L.Fabric(devices=1, precision="bf16-true")

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP)

    config = Config.from_name(model_name)
    print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module():
        og_model = model = GPT(config)
    # tinyllama support
    if not hasattr(model, "max_seq_length"):
        model.max_seq_length = model.config.block_size
    print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    if compile == "inductor":
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead", dynamic=dynamic)
    elif compile == "thunder":
        import thunder
        from thunder.executors.utils import Executor

        executors = [Executor.TORCH]
        executors = [Executor.NVFUSER] + executors
        model = thunder.compile(model, use_cudagraphs=False, executors_list=executors)
        model.max_seq_length = og_model.max_seq_length
    elif compile != "eager":
        raise ValueError(compile)

    model = fabric.setup(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-1, foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)

    train_data = Dataset(data_dir / "train.bin", model.max_seq_length, dynamic)
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


class Dataset(IterableDataset):
    def __init__(self, data_file: Path, max_seq_length: int, dynamic: bool):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = max_seq_length
        self.dynamic = dynamic

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            if self.dynamic:
                t = torch.randint(10, self.max_seq_length + 1, (1,))
            else:
                t = self.max_seq_length
            i = torch.randint(len(data) - t, (1,)).item()
            x = torch.from_numpy((data[i : i + t]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + t]).astype(np.int64))
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
