#!/usr/bin/env python3
"""
Fix eval_log.txt param header lines under aux_eval_logs.

Path structure: aux_eval_logs/XPR_NAME/ENV_NAME/ALG_ID/PARAM_ID_1=VAL_1/.../PARAM_ID_N=VAL_N/eval_log.txt

Replaces the single line:
  PARAM_ID_1,PARAM_ID_2,...,PARAM_ID_N
with:
  alg_id,PARAM_ID_1,PARAM_ID_2,...,PARAM_ID_N
  ALG_ID,VAL_1,VAL_2,...,VAL_N
"""

import argparse
from pathlib import Path


def parse_path(log_path: Path, base: Path) -> tuple[str, list[str], list[str]]:
    """Extract alg_id and (param_ids, param_vals) from path relative to base."""
    try:
        rel = log_path.relative_to(base)
    except ValueError:
        return None, [], []
    parts = rel.parts
    # parts = (xpr_name, env_name, alg_id, param1=val1, ..., eval_log.txt)
    if len(parts) < 4 or parts[-1] != "eval_log.txt":
        return None, [], []
    alg_id = parts[2]
    param_pairs = parts[3:-1]
    param_ids = []
    param_vals = []
    for p in param_pairs:
        if "=" not in p:
            return None, [], []
        idx = p.index("=")
        param_ids.append(p[:idx])
        param_vals.append(p[idx + 1 :])
    return alg_id, param_ids, param_vals


def fix_eval_log(log_path: Path, base: Path, dry_run: bool = False) -> bool:
    """Replace param header line in eval_log.txt. Returns True if file was modified."""
    alg_id, param_ids, param_vals = parse_path(log_path, base)
    if not param_ids:
        return False

    expected_header = ",".join(param_ids)
    new_header = "alg_id," + expected_header
    new_data_line = alg_id + "," + ",".join(param_vals)

    text = log_path.read_text()
    lines = text.splitlines(keepends=True)

    changed = False
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match line that is exactly the param header (strip newline for comparison)
        content = line.rstrip("\n\r")
        if content == expected_header and not content.startswith("alg_id,"):
            # Replace single line with header + data line
            new_lines.append(new_header + "\n")
            new_lines.append(new_data_line + ("\n" if not line.endswith("\n") else line[-1]))
            changed = True
            i += 1
            continue
        new_lines.append(line)
        i += 1

    if changed and not dry_run:
        log_path.write_text("".join(new_lines))
    return changed


def main():
    parser = argparse.ArgumentParser(description="Fix eval_log.txt param headers under aux_eval_logs")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="aux_eval_logs",
        help="Base directory to search (default: aux_eval_logs)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only print what would be changed",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    if not base.is_dir():
        print(f"Error: {base} is not a directory")
        return 1

    log_files = list(base.rglob("eval_log.txt"))
    fixed = 0
    skipped = 0
    for log_path in sorted(log_files):
        if fix_eval_log(log_path, base, dry_run=args.dry_run):
            fixed += 1
            print("Fixed:", log_path)
        else:
            skipped += 1

    print(f"\nDone: {fixed} fixed, {len(log_files) - fixed} skipped (no change or already correct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
