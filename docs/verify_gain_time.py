#!/usr/bin/env python3
"""Compare gain_time.csv outputs from the Python and Rust CLIs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class GainTimeRow:
    gain_m: float
    min_time_s: float
    avg_rate_m_per_hr: float
    note: str


def _load_rows(path: Path) -> List[GainTimeRow]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"gain_m", "min_time_s", "avg_rate_m_per_hr"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows: List[GainTimeRow] = []
        for raw in reader:
            try:
                gain = float(raw["gain_m"])
                min_time = float(raw["min_time_s"])
                rate = float(raw["avg_rate_m_per_hr"])
            except ValueError as exc:  # pragma: no cover - CSV parse errors
                raise ValueError(f"{path}: failed to parse numeric fields in row {raw}") from exc
            note = (raw.get("note") or "").strip()
            rows.append(GainTimeRow(gain_m=gain, min_time_s=min_time, avg_rate_m_per_hr=rate, note=note))
    return rows


def _keyed(rows: Iterable[GainTimeRow]) -> Dict[float, GainTimeRow]:
    lookup: Dict[float, GainTimeRow] = {}
    for row in rows:
        key = round(row.gain_m, 6)
        lookup[key] = row
    return lookup


def _summarise(py_rows: Dict[float, GainTimeRow], rs_rows: Dict[float, GainTimeRow]) -> Tuple[List[str], float, float, List[float]]:
    header = ["gain_m", "python_min_s", "rust_min_s", "delta_s", "python_rate", "rust_rate", "delta_rate"]
    lines: List[str] = []
    max_time_delta = 0.0
    max_rate_delta = 0.0
    missing_keys: List[float] = []
    for key in sorted(set(py_rows) | set(rs_rows)):
        py = py_rows.get(key)
        rs = rs_rows.get(key)
        if py is None or rs is None:
            missing_keys.append(key)
            continue
        dt = py.min_time_s - rs.min_time_s
        dr = py.avg_rate_m_per_hr - rs.avg_rate_m_per_hr
        max_time_delta = max(max_time_delta, abs(dt))
        max_rate_delta = max(max_rate_delta, abs(dr))
        lines.append(
            f"{py.gain_m:8.1f} {py.min_time_s:12.3f} {rs.min_time_s:12.3f} {dt:9.3f} {py.avg_rate_m_per_hr:12.3f} {rs.avg_rate_m_per_hr:12.3f} {dr:11.3f}"
        )
    return lines, max_time_delta, max_rate_delta, missing_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_csv", type=Path, help="gain_time.csv produced by the Python CLI")
    parser.add_argument("rust_csv", type=Path, help="gain_time.csv produced by the Rust CLI")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Allowed absolute difference in seconds between the two outputs (default: 1.0)",
    )
    args = parser.parse_args()

    py_rows = _keyed(_load_rows(args.python_csv))
    rs_rows = _keyed(_load_rows(args.rust_csv))

    lines, max_dt, max_dr, missing = _summarise(py_rows, rs_rows)

    print("gain_m   python_min_s  rust_min_s   delta_s  python_rate  rust_rate  delta_rate")
    for line in lines:
        print(line)

    if missing:
        missing_str = ", ".join(f"{val:.1f}" for val in missing)
        print(f"\nMissing gains in one of the inputs: {missing_str}")

    print(f"\nMax |delta_s| = {max_dt:.3f} s, Max |delta_rate| = {max_dr:.3f} m/h")
    if max_dt > args.tolerance:
        print(f"WARNING: time delta exceeds tolerance ({args.tolerance:.3f} s)")


if __name__ == "__main__":
    main()
