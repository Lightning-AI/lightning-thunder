import random
import string
import time

from benchmark_scripts import Target

from absl import logging, flags
from absl.testing import absltest

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

from thunder.tests.litgpt_model import Config, GPT
from transformers import AutoTokenizer

flags.DEFINE_string("compile", default="eager", help="Specify compile option: thunder|inductor|eager")

flags.DEFINE_integer("input_len", default=22_000, help="Specify input sequence length")
flags.DEFINE_integer("batch_size", default=1, help="Specify batch size")

flags.DEFINE_integer("max_iters", default=100, help="Specify number of training iterations")
flags.DEFINE_integer("warmup_iters", default=20, help="Specify number of warmup iterations")

flags = flags.FLAGS


class SecondsPerIter:
    def __init__(self, average_iters: int = 10) -> None:
        super().__init__()

        self.last_call = None
        self.count_calls = -1

        self.average_iters = average_iters
        self.last_iters = torch.zeros(average_iters)

    def add_batch(self, start_time: int = None) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter_ns()

        if start_time:
            self.last_call = start_time

        if self.last_call is not None:
            te = now - self.last_call
            self.last_iters[self.count_calls % self.average_iters] = te
        self.last_call = now
        self.count_calls += 1

    def compute(self) -> float:
        if self.count_calls == 0:
            return 0.0
        return self.last_iters[: self.count_calls].mean().item()


class DummyIterableDataset(IterableDataset):
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    def __iter__(self):
        while True:
            input_text = "".join(random.choices(string.ascii_letters + string.digits, k=flags.input_len))
            encoded = self._tokenizer(
                input_text, return_token_type_ids=True, max_length=flags.input_len, padding="max_length"
            )
            for k, v in encoded.items():
                encoded[k] = torch.tensor(v)

            yield encoded


def shift_targets_to_left(targets: torch.Tensor) -> torch.Tensor:
    targets = torch.roll(targets, dims=-1, shifts=-1)
    targets[:, -1] = 0
    return targets


def get_model(vocab_size: int) -> GPT:
    config = Config.from_name("Mistral-7B-v0.1")
    config.block_size = 32768
    config.padded_vocab_size = vocab_size

    return GPT(config)


def setup_compile(model) -> GPT:
    logging.info(f"Setting up compile option: {flags.compile}")
    if flags.compile == "thunder":
        import thunder

        return thunder.jit(model)
    elif flags.compile == "inductor":
        return torch.compile(model)
    else:
        return model


class MistralBenchmark(Target):
    def setUp(self):
        super().setUp()
        self.report_flag("input_len", flags.input_len)
        self.report_flag("batch_size", flags.batch_size)
        self.report_flag("max_iters", flags.max_iters)
        self.report_flag("warmup_iters", flags.warmup_iters)
        self.report_flag("compile", flags.compile)

        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        vocab_size = len(tokenizer.get_vocab())

        model = get_model(vocab_size)
        model.to(dtype=torch.bfloat16)
        model.to(device=self.device)
        self.model = setup_compile(model)

        self.mixed_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        logging.info(f"Running on {self.device}")

        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.995),
            eps=1e-6,
            weight_decay=1e-3,
        )

        train_dataset = DummyIterableDataset(tokenizer)
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=1)

        self.metric = SecondsPerIter(average_iters=10)

    def tearDown(self) -> None:
        self.report_metrics(
            {"iter_time": self.metric.compute(), "max_memory_allocated": torch.cuda.max_memory_allocated() / 1e9}
        )
        super().tearDown()

    def test_training(self) -> None:
        self.model.train()
        train_iter = iter(self.train_dataloader)
        logging.info("Started training")

        for step in range(flags.max_iters):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(self.device)

            with self.mixed_ctx:
                logits = self.model(input_ids)

            vocab_size = logits.size(-1)
            logits = logits.view(-1, vocab_size)

            targets = shift_targets_to_left(input_ids).view(-1)
            targets = targets.to(self.device)

            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=0)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if step > flags.warmup_iters:
                self.metric.add_batch()

        logging.info(f"Iter {step} with {self.metric.compute()*1e-6:.4f} ms per iteration.")


if __name__ == "__main__":
    absltest.main()
