import glob
import json
import sys
import time
from datetime import datetime

from lightning_sdk import Studio, Job, Machine, Status


def main(gh_run_id: str = ""):
    if not gh_run_id:
        gh_run_id = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    print("Creating studio...")
    s = Studio(f"thunder-benchmark-run{gh_run_id}", "oss-thunder", org="lightning-ai", create_ok=True)

    print("Uploading package and benchmark script...")
    s.upload_folder("dist", remote_path="dist")
    pkg_path = glob.glob("dist/*.whl")[0]
    s.upload_file("thunder/benchmarks/benchmark_hf.py", remote_path="benchmarks/benchmark_hf.py")

    print("Starting studio...")
    s.start()
    print("Installing Thunder and dependencies...")
    s.run(f"pip install {pkg_path} -U transformers 'numpy<2.0' 'nvfuser_cu128_torch27==0.2.27.dev20250501' 'nvidia-cudnn-frontend'")

    print("Running HF benchmark script...")
    job = Job.run(
        name=f"benchmark-run{gh_run_id}",
        command="pip list && python benchmarks/benchmark_hf.py",
        studio=s,
        machine=Machine.L40S,
        interruptible=True,
    )

    print("Stopping studio...")
    s.stop()

    print("Waiting for job to finish...")
    job.wait()
    status = str(job.status).lower()
    print(f"[{job.status}]\t {job.name}")

    report = {"benchmark_hf.py": status}
    with open("benchmark_hf_report.json", "w") as fp:
        json.dump(report, fp, indent=4)

    if job.status != Status.Completed:
        print("=" * 80)
        print(f"===== benchmark_hf.py -> {job.status} =====")
        print("=" * 80)
        print(job.logs)
        print("=" * 80)
        time.sleep(3)
        raise RuntimeError(f"Benchmark HF job {job.status}")
    # clean up
    job.delete()
    s.delete()


if __name__ == "__main__":
    # parse command line arguments
    args = sys.argv[1:]
    main(*args)
