import argparse
import os
import subprocess


class Runner:
    def __init__(self, command_and_args: list[str], verbose: bool):
        self._command_and_args = command_and_args
        self._verbose = verbose

    def env_var_for_fuel(self) -> str:
        return "NVFUSER_OPTIMIZATION_FUEL"

    def run(self, fuel: int) -> int:
        os.environ[self.env_var_for_fuel()] = str(fuel)
        print(f"Running {self._command_and_args} with {fuel} units of fuel...")
        result = subprocess.run(
            self._command_and_args,
            check=False,
            stdout=(None if self._verbose else subprocess.DEVNULL),
            stderr=(None if self._verbose else subprocess.DEVNULL),
        )
        print(f"Command returned {result.returncode} with {fuel} units of fuel.")
        return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
A script that bisects nvFuser's optimization fuel to isolate a compiler bug.

See `thunder.extend.FusionExecutor.get_fuel/set_fuel` for what optimization fuel is and how it is implemented.

This script finds the minimum unit of optimization fuel between `lower_bound` and `upper_bound` that causes `command_and_args` to fail (i.e. exit with non-zero). The user will then run `NVFUSER_OPTIMIZATION_FUEL=<minimum unit> <command_and_args>` as a minimal reproducer to identify the nvFusion that triggers the failure. Likely, the last nvFusion generated is the culprit.
    """
    )
    parser.add_argument("lower_bound", type=int, help="the lower bound for bisecting")
    parser.add_argument("upper_bound", type=int, help="the upper bound for bisecting")
    parser.add_argument("command_and_args", type=str, nargs=argparse.REMAINDER, help="the command and args to run")
    parser.add_argument("--verbose", action="store_true", help="whether to show stdout/stderr of the subprocess")
    args = parser.parse_args()

    runner = Runner(args.command_and_args, args.verbose)

    # The cornercases are not needed for the correctness of binary search. They are for catching errors earlier when the user specified a wrong lower/upper bound.
    if (exitcode := runner.run(args.lower_bound)) != 0:
        print("No need to bisect. Command failed with the lower bound.")
        exit(0)

    if (exitcode := runner.run(args.upper_bound)) == 0:
        print("Bisecting failed. Command passed with the upper bound.")
        exit(1)

    # Find the smallest fuel that fails `command_and_args`.
    low = args.lower_bound + 1  # +1 because we know `lower_bound` passed.
    high = args.upper_bound
    while low < high:
        mid = (low + high) // 2
        exitcode = runner.run(mid)
        if exitcode == 0:
            low = mid + 1
        else:
            high = mid
    assert low == high

    print(f"Bisecting succeeded. Run the following command as a minimal reproducer:")
    print(f"    {runner.env_var_for_fuel()}={low} {' '.join(args.command_and_args)}")
    print(f"The last nvFusion likely triggered the failure.")
