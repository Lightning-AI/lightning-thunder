import glob
import json
import os.path
import sys
from datetime import datetime

from lightning_sdk import Job, Machine, Status, Studio


def main(gh_run_id: str = ""):
    if not gh_run_id:
        gh_run_id = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    print("Creating studio...")
    s = Studio(f"thunder-transformer-engine-tests-run{gh_run_id}", "oss-thunder", org="lightning-ai", create_ok=True)
    print("Uploading package and scripts...")
    s.upload_folder("dist", remote_path="dist")
    s.upload_folder("requirements", remote_path="requirements")
    s.upload_folder("thunder/tests", remote_path="tests")
    pkg_path = glob.glob("dist/*.whl")[0]

    print("Starting studio...")
    s.start(machine=Machine.L40S, interruptible=False)
    print("Installing Thunder and other requirements...")
    s.run(f"pip install {pkg_path} -U -r requirements/test.txt")
    s.run("pip install --no-build-isolation 'transformer_engine[pytorch]'")

    # Define test commands
    test_commands = [
        "pytest tests/test_transformer_engine_executor.py -v -rs",
        "pytest tests/test_transformer_engine_v2_executor.py -v -rs",
        # TODO: Add DDP and FSDP tests
        # "pytest tests/distributed/test_ddp.py -k transformer -v -rs",
        # "pytest tests/distributed/test_fsdp.py -k transformer -v -rs",
    ]

    print("Running transformer_engine tests...")

    cmd_exit_code = {}
    for test in test_commands:
        output, exit_code = s.run_with_exit_code(test)
        print(output)
        cmd_exit_code[test] = (output, exit_code)

    print("Stopping studio...")
    s.stop()

    print("Cleaning up...")
    s.delete()

    print("Test Outputs:")
    for cmd, (output, exit_code) in cmd_exit_code.items():
        if exit_code != 0:
            print(f"Test {cmd} failed with exit code {exit_code}")

    if any(exit_code != 0 for output, exit_code in cmd_exit_code.values()):
        assert False, "Some tests failed"


if __name__ == "__main__":
    # parse command line arguments
    args = sys.argv[1:]
    main(*args)
