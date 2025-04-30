from datetime import datetime
import glob
import os.path

from lightning_sdk import Studio, Job, Machine, Status


def main():
    print("Creating studio...")
    s = Studio("thunder-quickstarts", "oss-thunder", org="lightning-ai", create_ok=True)
    print("Uploading package and scripts...")
    s.upload_folder("dist", remote_path="dist")
    s.upload_folder("examples/quickstart", remote_path="quickstart")

    print("Starting studio...")
    s.start()
    print("Installing Thunder and other requirements...")
    s.run("pip install lightning-thunder transformers -f dist/ -U")

    ls_quickstart = glob.glob("examples/quickstart/*.py")
    print("Found quickstart scripts:", ls_quickstart)

    print("Running quickstart scripts...")
    timestamp = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    jobs = [
        Job.run(
            name=f"ci-{timestamp}_{script}",
            command=f"python quickstart/{os.path.basename(script)}",
            studio=s,
            machine=Machine.L40S,
            interruptible=True,
        )
        for script in ls_quickstart
    ]

    print("Stopping studio...")
    s.stop()

    print("Waiting for jobs to finish...")
    failures = {}
    for i, job in enumerate(jobs):
        print(f"Waiting for {i}/{len(jobs)}: {job.name}")
        job.wait()
        if job.status == Status.Completed:
            print(f"Job '{job.name}' finished: {job.status}")
        else:
            failures[job.name] = job.status
            print(f"Job '{job.name}' {job.status}:\n{job.logs}")

    assert len(failures) == 0, f"Some jobs failed: {failures}"

    # print("Cleaning up...")
    # s.delete()


if __name__ == "__main__":
    main()
