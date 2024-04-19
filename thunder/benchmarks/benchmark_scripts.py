import json
import os
import subprocess

from typing import Any
from datetime import datetime
from functools import partial

from absl import flags
from absl import logging
from absl.testing import absltest, parameterized

flags.DEFINE_string("output_dir", default="./benchmark_results", help="Output directory for benchmark JSONs")


class Target(absltest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._failed_tests = []

        for fn_name in dir(self):
            if fn_name.startswith("test"):
                fn = getattr(self, fn_name)
                wrapped_fn = partial(self._test_wrapper, fn=fn)
                setattr(self, fn_name, wrapped_fn)

    # Add timestamp and defer exceptions so we can save the test status.
    def _test_wrapper(self, *args, fn=None, **kwargs):
        self._start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            return fn(*args, **kwargs)
        except Exception as ex:
            self._failed_tests.append(ex)

    def _get_git_revision(self) -> str:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd).decode("ascii").strip()

    def _report_benchmark_results(self):
        out_dir = flags.FLAGS.output_dir
        filename = os.path.join(out_dir, f"{self._name}_{self._start_time}.json")
        git_revision = self._get_git_revision()

        with open(filename, mode="w") as json_report:
            json.dump(
                {
                    "test_class": self._name,
                    "timestamp": self._start_time,
                    "head": git_revision,
                    "succeeded": not self.has_failed_tests(),
                    **self._reported_metrics,
                    **self._flags,
                },
                json_report,
            )

    def has_failed_tests(self):
        return len(self._failed_tests) > 0

    def setUp(self) -> None:
        super().setUp()
        self._name = type(self).__name__
        self._reported_metrics: dict = {}
        self._flags: dict = {}

    def tearDown(self) -> None:
        super().tearDown()
        self._report_benchmark_results()
        for exception in self._failed_tests:
            raise exception

    def report_flag(self, name: str, value: Any) -> None:
        self._flags[name] = value

    def report_metrics(self, metrics: dict[str, Any]) -> None:
        self._reported_metrics.update(metrics)


class Benchmarks(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._summary_metrics = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        if flags.FLAGS.output_dir:
            os.makedirs(flags.FLAGS.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()

    def run_standalone_script(self, filename: str, kwargs: dict[str, Any]) -> None:
        cmd_string = ["python", f"{filename}"]
        for key, val in kwargs.items():
            cmd_string.append("--" + str(key) + "=" + str(val))

        logging.info(f'Running: {" ".join(cmd_string)!r}')
        self.assertCommandSucceeds(cmd_string, env=os.environ)

    @parameterized.product(
        compile=(
            "eager",
            "inductor",
            "thunder",
        ),
        input_len=(4_096,),
        batch_size=(1,),
    )
    def test_mistral(self, **kwargs):
        self.run_standalone_script("mistral.py", kwargs)


if __name__ == "__main__":
    absltest.main()
