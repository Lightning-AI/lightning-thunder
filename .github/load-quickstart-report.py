import json
import sys
from datetime import datetime
import glob
import os.path

from lightning_sdk import Studio, Job, Machine, Status


def main(report_path: str = "quickstart_report.json"):
    """Load and print the quickstart report."""
    with open(report_path) as fp:
        report = json.load(fp)

    print("Quickstart report:")
    for name, status in report.items():
        status_icon = "✅" if status == "completed" else "❌"
        print(f"{status_icon}: {name}")


if __name__ == "__main__":
    # optional path to the report file
    system_args = sys.argv[1:]
    main_args = {"report_path": system_args[0]} if len(system_args) > 0 else {}
    main(**main_args)
