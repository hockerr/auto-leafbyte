"""Add dry-mass based traits (SLA, LMA) to a finished run.

SLA (specific leaf area, cm²/g) and LMA (leaf mass per area, g/m²) cannot come
from a photo alone, so they are computed once you supply the oven-dry mass per
sample.

Input CSV: any of these column names for the file and the mass are accepted::

    file,dry_mass_mg
    Herbivory Data 140盐肤木-1 (2026.07.08 17_50_19).png,123.4
    140盐肤木-2,98.7          # short label (the part before the parenthesis) also works

Usage::

    python tools/attach_dry_mass.py --run out/yanfumu --masses masses.csv
    # -> rewrites out/yanfumu/leaf_traits.csv with sla_cm2_g / lma_g_m2 filled
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leafbyte import traits  # noqa: E402

FILE_KEYS = ("file", "filename", "文件", "文件名", "sample", "样本")
MASS_KEYS = ("dry_mass_mg", "mass_mg", "干重", "干重_mg", "dry_mass")


def _read_masses(path: Path) -> dict[str, float]:
    masses: dict[str, float] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        file_key = next((headers[key] for key in FILE_KEYS if key in headers), None)
        mass_key = next((headers[key] for key in MASS_KEYS if key in headers), None)
        if file_key is None or mass_key is None:
            raise SystemExit(f"need a file column and a mass column in {path}; "
                             f"found {reader.fieldnames}")
        for row in reader:
            name = (row.get(file_key) or "").strip()
            raw = (row.get(mass_key) or "").strip()
            if not name or not raw:
                continue
            try:
                masses[name] = float(raw)
            except ValueError:
                continue
    return masses


def _lookup(masses: dict[str, float], filename: str) -> float | None:
    if filename in masses:
        return masses[filename]
    stem = Path(filename).stem
    for key in (stem, stem.split("(")[0].strip()):
        if key in masses:
            return masses[key]
    for key, value in masses.items():
        if key and (key in filename or key in stem):
            return value
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder (contains leaf_traits.csv)")
    parser.add_argument("--masses", required=True, help="CSV with file + dry mass (mg)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    traits_path = run_dir / "leaf_traits.csv"
    if not traits_path.exists():
        print(f"error: {traits_path} not found - run the pipeline first")
        return 1
    masses = _read_masses(Path(args.masses))

    with open(traits_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else ["file"]
    for column in traits.MASS_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    matched = 0
    for row in rows:
        mass = _lookup(masses, row.get("file", ""))
        record = {"traits": {key: _float(row.get(key)) for key in traits.TRAIT_COLUMNS},
                  "total_cm2": _float(row.get("intact_area_cm2"))}
        traits.attach_dry_mass(record, mass)
        for column in traits.MASS_COLUMNS:
            row[column] = record.get(column)
        matched += 1 if mass else 0

    with open(traits_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"matched dry mass for {matched}/{len(rows)} samples")
    print(f"updated: {traits_path}")
    missing = [row.get("file") for row in rows if not row.get("dry_mass_mg")]
    if missing:
        print(f"no mass found for: {', '.join(str(name) for name in missing[:5])}"
              + (" ..." if len(missing) > 5 else ""))
    return 0


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
