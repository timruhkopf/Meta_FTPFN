"""Collect `run_pairs.py`'s per-pair JSON shards into one dataframe.

Reads whatever is on disk right now -- safe to call while `run_pairs.py` is
still running elsewhere ("adding to it as we collect them"): a partially
populated results directory just yields a partial dataframe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_results(results_dir: str, model: str | None = None) -> pd.DataFrame:
    base = Path(results_dir)
    model_dirs = [base / model] if model else [p for p in base.iterdir() if p.is_dir()]

    rows = []
    for model_dir in model_dirs:
        for shard in sorted(model_dir.glob("*.json")):
            payload = json.loads(shard.read_text())
            row = {k: v for k, v in payload.items() if k not in ("ladder_r2", "fidelity")}
            for rung, r2 in payload.get("ladder_r2", {}).items():
                row[f"r2_{rung}"] = r2
            fid = payload.get("fidelity")
            if fid:
                row.update({f"fidelity_{k}": v for k, v in fid.items()})
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None, help="Optional path to write the aggregated parquet to")
    args = parser.parse_args()

    df = load_results(args.results_dir, args.model)
    print(df.describe(include="all"))
    if args.out:
        df.to_parquet(args.out)
        print(f"wrote {len(df)} rows to {args.out}")
