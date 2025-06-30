import sys
from datetime import datetime
import glob
import os

from lightning_sdk import Studio, Machine


def main(gh_run_id: str = ""):
    if not gh_run_id:
        gh_run_id = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")

    filenames = []
    chunk_size = 16
    with open("examples/coverage/all.txt") as f:
        lines = [el for el in f.readlines()]
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
        for i, chunk_lines in enumerate(chunks):
            filename = f"{i:03d}.txt"
            with open(f"examples/coverage/{filename}", "w") as f:
                f.writelines([el + "\n" for el in chunk_lines])
            filenames.append(filename)

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

    with open("examples/coverage/all.txt") as f:
        lines = [el for el in f.readlines()]

    hf_token = os.environ["HF_TOKEN"]
    print("Running thunder.jit coverage...")
    for filename in filenames:
        out = s.run(
            f"HF_TOKEN={hf_token} python coverage/jit_coverage_hf.py --models-file coverage/{filename} --output-dir data"
        )
        print(f"Processing {filename}")
        print(out)

    out = s.run("python coverage/jit_coverage_hf.py --output-dir data --results-file data.json")
    print(out)

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
