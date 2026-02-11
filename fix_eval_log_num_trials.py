#!/usr/bin/env python3
"""
Fix num_trials in eval_log.txt Evals CSV: set num_trials (column 4) to the value
of search_budget_consumed (column 6) for each data row.

Columns: run_idx,eval,eval_std,num_trials,runtime,search_budget_consumed,num_eval_samples
         0        1    2        3         4       5                      6
"""

import argparse
from pathlib import Path

EVAL_HEADER = "run_idx,eval,eval_std,num_trials,runtime,search_budget_consumed,num_eval_samples"
NUM_TRIALS_IDX = 3
SEARCH_BUDGET_IDX = 5
EXPECTED_NUM_FIELDS = 7


def fix_eval_log(log_path: Path, dry_run: bool = False) -> bool:
    """Set num_trials = search_budget_consumed for each Evals data row. Returns True if modified."""
    text = log_path.read_text()
    lines = text.splitlines(keepends=True)

    in_evals = False
    changed = False
    new_lines = []

    for line in lines:
        content = line.rstrip("\n\r")
        if content == EVAL_HEADER:
            in_evals = True
            new_lines.append(line)
            continue
        if in_evals:
            parts = content.split(",")
            if len(parts) == EXPECTED_NUM_FIELDS:
                if parts[NUM_TRIALS_IDX] != parts[SEARCH_BUDGET_IDX]:
                    parts[NUM_TRIALS_IDX] = parts[SEARCH_BUDGET_IDX]
                    new_lines.append(",".join(parts) + ("\n" if line.endswith("\n") else line[-1]))
                    changed = True
                else:
                    new_lines.append(line)
            else:
                # No longer in the Evals data block (wrong number of fields)
                in_evals = False
                new_lines.append(line)
        else:
            new_lines.append(line)

    if changed and not dry_run:
        log_path.write_text("".join(new_lines))
    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Fix num_trials column from search_budget_consumed in eval_log.txt Evals CSV"
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="aux_eval_logs",
        help="Base directory to search (default: aux_eval_logs)",
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="Only print what would be changed")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    if not base.is_dir():
        print(f"Error: {base} is not a directory")
        return 1

    log_files = list(base.rglob("eval_log.txt"))
    fixed = 0
    for log_path in sorted(log_files):
        if fix_eval_log(log_path, dry_run=args.dry_run):
            fixed += 1
            print("Fixed:", log_path)

    print(f"\nDone: {fixed} fixed, {len(log_files) - fixed} skipped (no change or already correct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
