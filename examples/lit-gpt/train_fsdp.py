import logging
import re
import time
from typing import Literal

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.utils.data import DataLoader, IterableDataset

from _fsdp_thunder import FSDPThunderStrategy
from thunder.tests.lit_gpt_model import GPT, Block, Config

model_name = "open_llama_3b"
learning_rate = 6e-4
micro_batch_size = 2
max_iters = 50


def main(
    compile: str = "eager", devices: int = 2, stage: str = "2", bucketing_strategy: Literal["NONE", "BLOCK"] = "NONE"
) -> None:
    fsdp_type = {"2": "ZERO2", "3": "ZERO3"}[stage]
    sharding_strategy = {"2": "SHARD_GRAD_OP", "3": "FULL_SHARD"}[stage]
    auto_wrap_policy = always_wrap_policy if bucketing_strategy.lower() == "none" else {Block}
    strategy = (
        FSDPThunderStrategy(sharding_strategy=fsdp_type, bucketing_strategy=bucketing_strategy)
        if compile == "thunder"
        else FSDPStrategy(auto_wrap_policy=auto_wrap_policy, sharding_strategy=sharding_strategy)
    )

    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-true")
    fabric.launch()

    fabric.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP)

    config = Config.from_name(model_name)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module():
        og_model = model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    if compile == "inductor":
        # Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
        pattern = re.compile(".*Profiler function .* will be ignored")
        logging.getLogger("torch._dynamo.variables.torch").addFilter(
            lambda record: not pattern.search(record.getMessage())
        )

        model = torch.compile(model)
    elif compile == "thunder":
        pass  # fabric.setup does this
    elif compile != "eager":
        raise ValueError(compile)

    model = fabric.setup(model)
    if compile == "thunder":
        model.max_seq_length = og_model.max_seq_length
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-1, foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)

    train_data = DummyDataset(model.max_seq_length)
    train_dataloader = DataLoader(train_data, batch_size=micro_batch_size, num_workers=2)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    train(fabric, model, optimizer, train_dataloader)
    fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


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
        fabric.print(f"iter {i}: loss {loss_item :.4f}, iter time: {(t1 - iter_t0) * 1000:.2f}ms")
    fabric.print(f"Total time: {(t1 - t0):.2f}s")


class DummyDataset(IterableDataset):
    def __init__(self, max_seq_length: int):
        super().__init__()
        self.max_seq_length = max_seq_length

    def __iter__(self):
        t = self.max_seq_length
        while True:
            data = torch.randint(0, 100, (t + 1,), dtype=torch.int64)
            x = data[:t]
            y = data[1 : t + 1]
            yield x, y


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(main)
