import json
import sys


def main(report_path: str = "jit_coverage_report.json"):
    """Load and print the jit coverage report."""
    with open(report_path) as fp:
        report = json.load(fp)

    success = [el for el in report if el["status"] == "[SUCCESS]"]
    skipped = [el for el in report if el["status"] == "[SKIPPED]"]
    failure = [el for el in report if el["status"] == "[FAILURE]"]

    for el in success:
        print(f"🟢 [SUCCESS] {el['model']}")

    for el in skipped:
        print(f"🟡 [SKIPPED] {el['model']}")

    for el in failure:
        print(f"⛔️ [FAILURE] {el['model']}")
        print(f"   Error:    {el['last']}")


if __name__ == "__main__":
    # optional path to the report file
    system_args = sys.argv[1:]
    main_args = {"report_path": system_args[0]} if len(system_args) > 0 else {}
    main(**main_args)
