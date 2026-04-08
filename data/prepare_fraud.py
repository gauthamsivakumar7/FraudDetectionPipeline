from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class FraudPrepConfig:
    raw_dir: Path
    """Single merged output: competition train rows (with labels) then test rows (isFraud empty)."""
    out_csv: Path
    join_key: str = "TransactionID"
    target_col: str = "isFraud"
    # If True, only keep identity columns that appear in both train and test identity tables
    align_identity_columns: bool = True
    chunksize: int = 50_000


_ID_DASH_RE = re.compile(r"^id-(\d{2})$")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize known header variations:
    - test_identity.csv sometimes uses id-01..id-38; train uses id_01..id_38
    """
    rename = {}
    for c in df.columns:
        m = _ID_DASH_RE.match(c)
        if m:
            rename[c] = f"id_{m.group(1)}"
    if rename:
        df = df.rename(columns=rename)
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    # `low_memory=False` makes dtype inference more consistent on wide CSVs.
    # Use pandas default CSV engine (fast).
    return pd.read_csv(path, low_memory=False)

def _read_csv_chunks(path: Path, chunksize: int):
    return pd.read_csv(path, low_memory=False, chunksize=chunksize)


def _read_columns(path: Path) -> list[str]:
    # Header-only read (fast, avoids loading the entire file).
    return list(pd.read_csv(path, nrows=0).columns)


def _load_identity(
    id_path: Path,
    join_key: str,
    *,
    identity_keep_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    ident = _standardize_columns(_read_csv(id_path))
    if join_key not in ident.columns:
        raise ValueError(f"{id_path.name} missing join key column {join_key!r}")
    if identity_keep_columns is not None:
        keep = [c for c in identity_keep_columns if c in ident.columns]
        ident = ident[[join_key] + keep]
    return ident


def _merge_write_chunks(
    tx_path: Path,
    ident: pd.DataFrame,
    out_path: Path,
    join_key: str,
    *,
    chunksize: int,
    append: bool = False,
    expected_columns: Optional[list[str]] = None,
    ensure_target_col: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Stream-merge transaction chunks with identity and write to CSV incrementally.
    If append=True, writes after existing file (no header row); aligns columns to expected_columns.
    If ensure_target_col is set and missing from a chunk, it is added as NA (e.g. competition test rows).
    Returns (rows_written, cols_written).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    cols_written = 0

    first = not append
    for i, chunk in enumerate(_read_csv_chunks(tx_path, chunksize=chunksize), start=1):
        if join_key not in chunk.columns:
            raise ValueError(f"{tx_path.name} missing join key column {join_key!r}")

        merged = chunk.merge(ident, on=join_key, how="left", suffixes=("", "_id"))
        if ensure_target_col and ensure_target_col not in merged.columns:
            merged[ensure_target_col] = pd.NA
        if expected_columns is not None:
            merged = merged.reindex(columns=expected_columns)

        mode = "w" if first else "a"
        header = first
        merged.to_csv(out_path, index=False, mode=mode, header=header)

        rows_written += len(merged)
        cols_written = merged.shape[1]
        first = False

        if i == 1 or i % 10 == 0:
            print(
                f"  - {tx_path.name}: wrote chunk {i} "
                f"(rows so far: {rows_written:,})",
                flush=True,
            )

    return rows_written, cols_written


def prepare_fraud(config: FraudPrepConfig) -> None:
    train_tx_path = config.raw_dir / "train_transaction.csv"
    train_id_path = config.raw_dir / "train_identity.csv"
    test_tx_path = config.raw_dir / "test_transaction.csv"
    test_id_path = config.raw_dir / "test_identity.csv"

    for p in (train_tx_path, train_id_path, test_tx_path, test_id_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # Optionally align identity columns across train/test so the merged outputs share schema.
    identity_keep: Optional[list[str]] = None
    if config.align_identity_columns:
        train_id_cols = _standardize_columns(pd.DataFrame(columns=_read_columns(train_id_path))).columns
        test_id_cols = _standardize_columns(pd.DataFrame(columns=_read_columns(test_id_path))).columns
        common = sorted(set(train_id_cols).intersection(set(test_id_cols)))
        common = [c for c in common if c != config.join_key]
        identity_keep = common

    # Identity tables are comparatively small; load fully once.
    print("Loading identity tables...", flush=True)
    train_ident = _load_identity(
        train_id_path,
        config.join_key,
        identity_keep_columns=identity_keep,
    )
    test_ident = _load_identity(
        test_id_path,
        config.join_key,
        identity_keep_columns=identity_keep,
    )
    print(f"  - train_identity shape: {train_ident.shape}", flush=True)
    print(f"  - test_identity  shape: {test_ident.shape}", flush=True)

    print("Merging + writing (streaming): competition train →", config.out_csv, flush=True)
    train_rows, train_cols = _merge_write_chunks(
        train_tx_path,
        train_ident,
        config.out_csv,
        config.join_key,
        chunksize=config.chunksize,
        append=False,
    )

    merged_header = _read_columns(config.out_csv)
    if config.target_col not in merged_header:
        raise ValueError(
            f"Expected training label column {config.target_col!r} in merged output. "
            f"Is it present in {train_tx_path.name}?"
        )

    print("Appending competition test rows (same columns; empty labels)...", flush=True)
    test_rows, test_cols = _merge_write_chunks(
        test_tx_path,
        test_ident,
        config.out_csv,
        config.join_key,
        chunksize=config.chunksize,
        append=True,
        expected_columns=merged_header,
        ensure_target_col=config.target_col,
    )

    print(
        f"Finished → {config.out_csv}\n"
        f"  Labeled (train) rows: {train_rows:,}, cols: {train_cols}\n"
        f"  Unlabeled (test) rows appended: {test_rows:,}, cols: {test_cols}\n"
        f"  Total rows: {train_rows + test_rows:,}",
        flush=True,
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare IEEE-CIS fraud dataset by merging transaction + identity tables."
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/fraud"),
        help="Directory containing train/test transaction/identity CSVs",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/processed/fraud_merged.csv"),
        help="Single output CSV: labeled train rows then unlabeled test rows (same schema)",
    )
    p.add_argument(
        "--join-key",
        type=str,
        default="TransactionID",
        help="Join key for merging identity onto transaction",
    )
    p.add_argument(
        "--target-col",
        type=str,
        default="isFraud",
        help="Training label column name (present only in train_transaction.csv)",
    )
    p.add_argument(
        "--no-align-identity-columns",
        action="store_true",
        help="If set, keep all identity columns even if train/test identity headers differ",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Rows per transaction chunk to merge/write (larger = faster, more RAM)",
    )
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    config = FraudPrepConfig(
        raw_dir=args.raw_dir,
        out_csv=args.out_csv,
        join_key=str(args.join_key),
        target_col=str(args.target_col),
        align_identity_columns=not bool(args.no_align_identity_columns),
        chunksize=int(args.chunksize),
    )

    prepare_fraud(config)


if __name__ == "__main__":
    main()

