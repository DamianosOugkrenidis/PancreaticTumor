import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten a run's training history into a CSV for plotting.")
    parser.add_argument("--run-dir", required=True, help="Run directory created by scripts/train.py.")
    parser.add_argument("--history", default=None,
                        help="History JSON (default: <run-dir>/artifacts/training_history.json).")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: <run-dir>/artifacts/training_history.csv).")
    return parser.parse_args()


def flatten_history(history: list) -> tuple[list[str], list[dict]]:
    rows = []
    for record in history:
        row = {"epoch": record.get("epoch")}
        for key, value in record.items():
            if key in {"epoch", "val"}:
                continue
            if isinstance(value, (int, float)):
                row[key] = value
        for key, value in (record.get("val") or {}).items():
            if isinstance(value, (int, float)):
                row[f"val_{key}"] = value
        rows.append(row)

    columns = ["epoch"]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns, rows


def main() -> int:
    arguments = parse_arguments()
    run_dir = Path(arguments.run_dir).resolve()
    history_path = Path(arguments.history) if arguments.history else \
        run_dir / "artifacts" / "training_history.json"
    if not history_path.is_file():
        raise SystemExit(f"History not found: {history_path}")

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload.get("history", payload)

    columns, rows = flatten_history(history)
    output_path = Path(arguments.output) if arguments.output else \
        run_dir / "artifacts" / "training_history.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
