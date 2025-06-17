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
    s.start()
    print("Installing Thunder and other requirements...")
    s.run(f"pip install {pkg_path} -U")

    # Define test commands
    test_commands = [
        "pytest thunder/tests/test_transformer_engine_executor.py -v",
        "pytest thunder/tests/distributed/test_ddp.py -k transformer -v -rs",
        "pytest thunder/tests/distributed/test_fsdp.py -k transformer -v -rs",
        "pytest thunder/tests/test_transformer_engine_v2_executor.py -v -rs"
    ]

    print("Running transformer tests...")
    jobs = {
        f"test_{i}": Job.run(
            name=f"ci-run{gh_run_id}_transformer_test_{i}",
            command=cmd,
            studio=s,
            machine=Machine.L40S,
            interruptible=True,
        )
        for i, cmd in enumerate(test_commands)
    }

    print("Stopping studio...")
    s.stop()

    print("Waiting for jobs to finish...")
    report, failures = {}, {}
    for name, job in jobs.items():
        job.wait()
        print(f"[{job.status}]\t {job.name}")
        report[name] = str(job.status).lower()
        if job.status != Status.Completed:
            failures[name] = job.logs
        else:  # clean up successful jobs
            job.delete()

    with open("transformer_tests_report.json", "w") as fp:
        json.dump(report, fp, indent=4)

    print("Showing logs of failed jobs...")
    separator = "=" * 80
    for name, logs in failures.items():
        offset = "=" * (80 - 5 - 2 - len(name))
        print(f"{separator}\n===== {name} {offset}\n{separator}")
        print(logs)
        print(separator + "\n" * 5)

    assert not failures

    print("Cleaning up...")
    s.delete()


if __name__ == "__main__":
    # parse command line arguments
    args = sys.argv[1:]
    main(*args) 