"""
Script to run all lit-GPT models available as a parametrized test using abseil's unittest framework.
Runs a parametrized product over all configs specified, compiler options, distributed modes etc.
Uses environment variables to modify default behavior
BENCHMARK_FILE - use this env variable to control the benchmarking script that is being targeted.
                 Runs the benchmark_litgpt.py under the installed thunder/benchmarks by default.
MID_BENCHMARK_OUT - use this env variable to control whether you want to see the combined results
                    between each test.
BENCHMARK_OUT_FORMAT - use this env variable to control the format in which the results are presented.
                    Uses 'xlsx' by default. Supported: 'none', 'print', 'xlsx'.
"""

import torch
from absl.testing import parameterized
from absl.testing import absltest
from collections import defaultdict
import os
import subprocess
import json
import pandas as pd
from datetime import datetime


class Runner:
    """
    Benchmark Runner class to
        a) Launch the training benchmarking run,
        b) Store results from all tests,
        c) Compile results as xlsx file
    """

    def __init__(self, benchmark_file, mid_benchmark_out, output_format):
        self.dataframe_data = []
        self.json_file_path = "/tmp/benchmark_litgpt_data.json"
        self.benchmark_file = benchmark_file
        self.mid_benchmark_out = mid_benchmark_out
        self.output_format = output_format

    def __enter__(self):
        return self

    def add_to_dataframe(self):
        if self.perf_metrics_dict:
            if (
                "tokens_per_sec_per_gpu" not in self.perf_metrics_dict.keys()
            ):  # In case of OutofMemory error, this is already marked 'OOM'
                self.perf_metrics_dict["tokens_per_sec_per_gpu"] = (
                    self.perf_metrics_dict["tokens_per_sec"] / self.perf_metrics_dict["Num GPUS"]
                )
            self.dataframe_data.append(self.perf_metrics_dict)

    def complete_dataframe(self, is_teardown):
        if not self.dataframe_data:
            # The benchmark probably failed
            return
        # Called when tearing down the parametrized test
        # This generates a summarized dataframe for each perf metric and saves as a xlsx file
        df = pd.DataFrame(self.dataframe_data)
        df["Sharding Size"] = df["Sharding Size"].fillna(
            "none"
        )  # Convert None Type to string so that pivot table can group.
        index_list = [
            "model_name",
            "Num GPUS",
            "Seq Len",
            "Micro BS",
            "Global BS",
            "GA",
            "Distributed Mode",
            "Sharding Size",
        ]

        self.iter_time_df = df.pivot_table(
            index=index_list, columns="compiler", values="average_iter_time", aggfunc="first"
        ).reset_index()
        self.tokens_per_sec_df = df.pivot_table(
            index=index_list, columns="compiler", values="tokens_per_sec", aggfunc="first"
        ).reset_index()
        self.tokens_per_sec_per_gpu_df = df.pivot_table(
            index=index_list, columns="compiler", values="tokens_per_sec_per_gpu", aggfunc="first"
        ).reset_index()
        self.memory_used_GB_df = df.pivot_table(
            index=index_list, columns="compiler", values="memory_used_GB", aggfunc="first"
        ).reset_index()

        if self.output_format == "xlsx":
            output_ext = {
                "xlsx": ".xlsx",
            }[self.output_format]
            if not is_teardown:
                filename = "mid_output_parameterized_results" + str(output_ext)
            else:
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
                filename = f"{current_time}_litgpt_benchmark" + str(output_ext)

            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                self.iter_time_df.to_excel(writer, sheet_name="Average Iter Time (ms)")
                self.tokens_per_sec_df.to_excel(writer, sheet_name="Tokens per sec")
                self.tokens_per_sec_per_gpu_df.to_excel(writer, sheet_name="Tokens per sec per GPU")
                self.memory_used_GB_df.to_excel(writer, sheet_name="Memory allocated GB")
        elif self.output_format == "print":
            print("\nAVERAGE ITERATION TIME (ms)")
            print(self.iter_time_df)
            print("\nTHROUGHPUT (tokens/s)")
            print(self.tokens_per_sec_df)
            print("\nNORMALIZED THROUGHPUT (tokens/s/GPU)")
            print(self.tokens_per_sec_per_gpu_df)
            print("\nMEMORY ALLOCATED (GB)")
            print(self.memory_used_GB_df)

    def run_benchmark(self, kwargs):
        command_list = []
        for key, val in kwargs.items():
            command_list.append("--" + str(key) + "=" + str(val))
        if kwargs["distributed_mode"] != "none":
            nproc_per_node = torch.cuda.device_count()
            subprocess_cmd = [
                "torchrun",
                f"--nproc_per_node={nproc_per_node}",
                "--nnodes=1",
                f"{self.benchmark_file}",
                "--return_metrics_as_json=True",
                f"--json_path={self.json_file_path}",
            ]
            subprocess_cmd.extend(command_list)
        else:
            subprocess_cmd = [
                "python",
                f"{self.benchmark_file}",
                "--return_metrics_as_json=True",
                f"--json_path={self.json_file_path}",
            ]
            subprocess_cmd.extend(command_list)

        print(f'Running {" ".join(subprocess_cmd)!r}')
        proc_output = subprocess.run(subprocess_cmd, capture_output=True, text=True)

        self.perf_metrics_dict = {}
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path) as file:
                self.perf_metrics_dict = json.load(file)
            # Cleanup after the benchmark finishes. It might have failed before creating this
            os.remove(self.json_file_path)

        if proc_output.returncode:
            if "CUDA out of memory" in proc_output.stdout or "CUDA error: out of memory" in proc_output.stderr:
                defaultdict_oom = defaultdict(lambda: "OOM")
                defaultdict_oom.update(self.perf_metrics_dict)
                self.perf_metrics_dict = defaultdict_oom
                pass_str = "TestCase did not finish reporting metrics due to CUDA out of memory error. Reporting OOM and triggering test success."
                return True, pass_str
            print(proc_output.stdout)
            print(proc_output.stderr)
            fail_str = "TestCase did not finish reporting metrics due to an unknown error. Triggering test failure."
            return False, fail_str
        return True, "Test passed successfully."


class Test(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def get_installed_thunder_path():
            import thunder

            thunder_init = thunder.__file__
            thunder_benchmark_file = str(thunder_init).replace("__init__.py", "benchmarks/benchmark_litgpt.py")
            return thunder_benchmark_file

        benchmark_file = os.getenv("BENCHMARK_FILE", get_installed_thunder_path())
        mid_benchmark_out = bool(os.getenv("MID_BENCHMARK_OUT", 0))
        output_format = str(os.getenv("BENCHMARK_OUT_FORMAT", "xlsx"))  # Can take none, print, xlsx as of 03/12
        cls.runner = Runner(
            benchmark_file=benchmark_file, mid_benchmark_out=mid_benchmark_out, output_format=output_format
        )

    @classmethod
    def tearDownClass(cls):
        cls.runner.complete_dataframe(is_teardown=True)
        super().tearDownClass()

    # @parameterized.product(
    #     (dict(distributed_mode = "fsdp", shard_mode = "zero2"),
    #      dict(distributed_mode = "fsdp", shard_mode = "zero3"),
    #      dict(distributed_mode = "ddp", shard_mode = "none"),
    #      dict(distributed_mode = "none", shard_mode = "none")),
    #     (dict(model_name = 'Llama-2-7b-hf', micro_batch_size=1),
    #      dict(model_name = 'Llama-2-7b-hf', micro_batch_size=2),
    #      dict(model_name = 'Llama-2-13b-hf', micro_batch_size=1),
    #      dict(model_name = 'Llama-2-13b-hf', micro_batch_size=2),
    #      dict(model_name = 'stablecode-completion-alpha-3b', micro_batch_size=1),
    #      dict(model_name = 'stablecode-completion-alpha-3b', micro_batch_size=2),
    #      dict(model_name = 'Mistral-7B-v0.1', micro_batch_size=1),
    #      dict(model_name = 'Mistral-7B-v0.1', micro_batch_size=2),
    #      dict(model_name = 'open_llama_3b', micro_batch_size=1),
    #      dict(model_name = 'open_llama_3b', micro_batch_size=2),
    #      dict(model_name = 'open_llama_3b', micro_batch_size=4),
    #      dict(model_name = 'open_llama_7b', micro_batch_size=1),
    #      dict(model_name = 'open_llama_7b', micro_batch_size=2),
    #      dict(model_name = 'open_llama_7b', micro_batch_size=4),
    #      dict(model_name = 'open_llama_13b', micro_batch_size=1),
    #      dict(model_name = 'open_llama_13b', micro_batch_size=2),
    #      dict(model_name = 'stablelm-base-alpha-3b', micro_batch_size=1),
    #      dict(model_name = 'stablelm-base-alpha-3b', micro_batch_size=2),
    #      dict(model_name = 'stablelm-base-alpha-3b', micro_batch_size=4),
    #      dict(model_name = 'stablelm-base-alpha-7b', micro_batch_size=1),
    #      dict(model_name = 'stablelm-base-alpha-7b', micro_batch_size=2),
    #      dict(model_name = 'pythia-2.8b', micro_batch_size=1),
    #      dict(model_name = 'pythia-2.8b', micro_batch_size=2),
    #      dict(model_name = 'pythia-2.8b', micro_batch_size=4),
    #      dict(model_name = 'pythia-6.9b', micro_batch_size=1),
    #      dict(model_name = 'pythia-6.9b', micro_batch_size=2),
    #      dict(model_name = 'pythia-12b', micro_batch_size=1),
    #      dict(model_name = 'pythia-12b', micro_batch_size=2),
    #      dict(model_name = 'falcon-7b', micro_batch_size=1),
    #      dict(model_name = 'falcon-7b', micro_batch_size=2)),
    #     compile = ("eager", "inductor", "thunder", "thunder_inductor",)
    # )

    @parameterized.product(
        distributed_mode=("fsdp",),
        shard_mode=("zero2",),
        model_name=("Llama-2-7b-hf",),
        micro_batch_size=(
            1,
            4,
        ),
        compile=(
            "eager",
            "inductor",
            "thunder",
            "thunder_inductor",
        ),
    )
    def test(self, **kwargs):
        kwargs["nsys_enabled"] = False
        kwargs["dynamic"] = False
        self.__file__ = __file__

        try:
            os.remove(self.runner.json_file_path)
        except FileNotFoundError:
            pass

        run_ok, run_msg = self.runner.run_benchmark(kwargs)
        if run_ok:
            print(run_msg)
            self.runner.add_to_dataframe()
            if self.runner.mid_benchmark_out:
                self.runner.complete_dataframe(is_teardown=False)
        else:
            self.fail(run_msg)


if __name__ == "__main__":
    absltest.main()
