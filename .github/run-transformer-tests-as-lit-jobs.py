import json
import sys
from datetime import datetime
import glob
import os.path

from lightning_sdk import Studio, Job, Machine, Status


def main(gh_run_id: str = ""):
    if not gh_run_id:
        gh_run_id = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    print("Creating studio...")
    s = Studio(f"thunder-transformer-tests-run{gh_run_id}", "oss-thunder", org="lightning-ai", create_ok=True)
    print("Uploading package and scripts...")
    s.upload_folder("dist", remote_path="dist")
    pkg_path = glob.glob("dist/*.whl")[0]

    print("Starting studio...")
    s.start(machine=Machine.L40S, interruptible=False)
    print("Installing Thunder and other requirements...")
    s.run(f"pip install {pkg_path} -U")
    s.run("pip install --no-build-isolation transformer_engine[pytorch]")

    # Define test commands
    test_commands = [
        "pytest thunder/tests/test_transformer_engine_executor.py -v",
        "pytest thunder/tests/distributed/test_ddp.py -k transformer -v -rs",
        "pytest thunder/tests/distributed/test_fsdp.py -k transformer -v -rs",
        "pytest thunder/tests/test_transformer_engine_v2_executor.py -v -rs",
    ]

    print("Running transformer tests...")

    test_outputs = []
    for test in test_commands:
        test_outputs.append(s.run(test))

    print("Stopping studio...")
    s.stop()

    print("Cleaning up...")
    s.delete()

    print("Test Outputs:")
    for test_output in test_outputs:
        print(test_output)


if __name__ == "__main__":
    # parse command line arguments
    args = sys.argv[1:]
    main(*args)
