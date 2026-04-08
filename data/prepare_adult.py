from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
from typing import Dict, Iterable, List, Optional, Tuple


ADULT_COLUMNS: List[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

ADULT_NUMERIC_COLUMNS = {
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
}


@dataclass(frozen=True)
class AdultPrepConfig:
    raw_dir: Path
    out_csv: Path
    drop_unknowns: bool = False
    target_col: str = "target"


def _iter_rows(path: Path) -> Iterable[List[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("|"):  # comment lines in adult.test
                continue
            yield [c.strip() for c in row]


def _normalize_value(v: str) -> str:
    v = v.strip()
    return "" if v == "?" else v


def _normalize_income_label(v: str) -> str:
    # `adult.test` has labels like "<=50K." / ">50K."
    v = v.strip()
    if v.endswith("."):
        v = v[:-1]
    return v


def _parse_int_or_empty(v: str) -> int | str:
    v = v.strip()
    if v == "":
        return ""
    try:
        return int(v)
    except ValueError:
        return ""


def _load_rows(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in _iter_rows(path):
        if len(row) != len(ADULT_COLUMNS):
            # Skip malformed rows rather than producing shifted columns.
            continue
        d: Dict[str, object] = {}
        for k, v in zip(ADULT_COLUMNS, row):
            v_norm = _normalize_value(v)
            if k in ADULT_NUMERIC_COLUMNS:
                d[k] = _parse_int_or_empty(v_norm)
            else:
                d[k] = v_norm
        d["income"] = _normalize_income_label(str(d["income"]))
        out.append(d)
    return out


def prepare_adult(config: AdultPrepConfig) -> Tuple[int, List[str]]:
    train_path = config.raw_dir / "adult.data"
    test_path = config.raw_dir / "adult.test"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")

    rows = _load_rows(train_path) + _load_rows(test_path)
    out_columns = [c for c in ADULT_COLUMNS if c != "income"] + [config.target_col]

    config.out_csv.parent.mkdir(parents=True, exist_ok=True)
    config.out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with config.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_columns)
        w.writeheader()
        for r in rows:
            if config.drop_unknowns and any(v == "" for k, v in r.items() if k != "income"):
                continue
            target = 1 if r["income"] == ">50K" else 0
            out_row = {k: r[k] for k in ADULT_COLUMNS if k != "income"}
            out_row[config.target_col] = target
            w.writerow(out_row)
            written += 1

    return written, out_columns


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare UCI Adult dataset into a clean CSV.")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/adult"),
        help="Directory containing adult.data and adult.test",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/adult.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--drop-unknowns",
        action="store_true",
        help='If set, drop rows containing any "?" unknowns',
    )
    p.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Name of the target column to create (0/1)",
    )
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    config = AdultPrepConfig(
        raw_dir=args.raw_dir,
        out_csv=args.out_csv,
        drop_unknowns=bool(args.drop_unknowns),
        target_col=str(args.target_col),
    )
    n, cols = prepare_adult(config)
    print(f"Wrote {n:,} rows → {config.out_csv}")
    print("Columns:", ", ".join(cols))


if __name__ == "__main__":
    main()
