import json
from datetime import datetime

from lightning_sdk import Studio, Job, Machine, Status


def main():
    print("Creating studio...")
    s = Studio("thunder-benchmark-hf", "oss-thunder", org="lightning-ai", create_ok=True)

    print("Uploading package and benchmark script...")
    s.upload_folder("dist", remote_path="dist")
    s.upload_file("thunder/benchmarks/benchmark_hf.py", remote_path="benchmarks/benchmark_hf.py")

    print("Starting studio...")
    s.start()
    print("Installing Thunder and dependencies...")

    # install from latest commit
    s.run(
        """
    git clone https://github.com/lightning-AI/lightning-thunder.git &&
    pip install -e lightning-thunder &&
    pip install transformers nvfuser_cu128_torch27==0.2.27.dev20250501
    """
    )

    print("Running HF benchmark script...")
    timestamp = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    job = Job.run(
        name=f"benchmark-hf-{timestamp}",
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
        print("===== benchmark_hf.py FAILED =====")
        print("=" * 80)
        print(job.logs)
        print("=" * 80)
        raise RuntimeError("Benchmark HF job failed")


if __name__ == "__main__":
    main()
