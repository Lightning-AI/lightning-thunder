import sys
from datetime import datetime
import glob
import os

from lightning_sdk import Studio, Machine


def main(gh_run_id: str = ""):
    if not gh_run_id:
        gh_run_id = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

    print("Creating studio...")
    s = Studio(f"thunder-jit-coverage-run{gh_run_id}", "oss-thunder", org="lightning-ai", create_ok=True)

    print("Uploading package and scripts...")
    s.upload_folder("dist", remote_path="dist")
    pkg_path = glob.glob("dist/*.whl")[0]
    s.upload_folder("examples/coverage", remote_path="coverage")

    print("Starting studio...")
    s.start(machine=Machine.L40S, interruptible=True)

    print("Installing Thunder and other requirements...")
    s.run(f"pip install {pkg_path} -U -r coverage/requirements.txt")

    hf_token = os.environ["HF_TOKEN"]
    print("Running thunder.jit coverage...")
    s.run(
        f"HF_TOKEN={hf_token} python coverage/jit_coverage_hf.py --models-file coverage/all.txt --output-dir data --results-file data.json"
    )

    data_json = s.run("cat data.json")

    with open("jit_coverage_report.json", "w") as f:
        f.write(data_json)

    print("Stopping studio...")
    s.stop()
    s.delete()


if __name__ == "__main__":
    # parse command line arguments
    args = sys.argv[1:]
    main(*args)
