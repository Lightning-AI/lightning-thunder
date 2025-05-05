import json
import sys


def main(report_path: str = "quickstart_report.json"):
    """Load and print the quickstart report."""
    with open(report_path) as fp:
        report = json.load(fp)

    success_count = sum(status == "completed" for status in report.values())
    overall_status = "🟢" if success_count == len(report) else "⛔"
    print(f"Quickstart report {overall_status} with {success_count} out of {len(report)} successful:")    # Sort so that entries with status "success" (or "completed") are last
    for name, status in sorted(report.items(), key=lambda x: x[1] == "completed"):
        status_icon = "✔️" if status == "completed" else "❌"
        print(f"{status_icon} {name}")


if __name__ == "__main__":
    # optional path to the report file
    system_args = sys.argv[1:]
    main_args = {"report_path": system_args[0]} if len(system_args) > 0 else {}
    main(**main_args)
