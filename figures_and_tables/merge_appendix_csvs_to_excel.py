#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_appendix_csvs_to_excel.py

Combine appendix CSV tables into ONE Excel file with TWO sheets:
  - Appendix1 : all Fig.5 tables
  - Appendix2 : all Fig.6 tables

CSV files are expected to be in the SAME directory as this script
(i.e. figures_and_tables/), inside subfolders:

  figures_and_tables/
      Appendix1/
          fig5_table_<MASK>_rate<RATE>.csv
      Appendix2/
          fig6_table_<MASK>_rate<RATE>.csv

Ordering inside each sheet:
  Masks  : MCAR, MAR, MNAR, MCAR_pair, MAR_pair, MNAR_pair
  Rates  : 5, 10, 15, 20, 25, 30

Each (mask, rate) block is stacked vertically with a header row.

Usage:
  cd figures_and_tables
  python merge_appendix_csvs_to_excel.py
"""

import os
import re
from typing import List

import pandas as pd

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MASK_ORDER: List[str] = ["MCAR", "MAR", "MNAR", "MCAR_pair", "MAR_pair", "MNAR_pair"]
RATE_ORDER: List[int] = [5, 10, 15, 20, 25, 30]

APPENDIX1_DIR = "Appendix1"
APPENDIX2_DIR = "Appendix2"

OUT_XLSX = "Appendix_Tables.xlsx"

PAT_FIG5 = re.compile(r"^fig5_table_(?P<mask>.+)_rate(?P<rate>\d+)\.csv$", re.IGNORECASE)
PAT_FIG6 = re.compile(r"^fig6_table_(?P<mask>.+)_rate(?P<rate>\d+)\.csv$", re.IGNORECASE)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _collect_blocks(folder: str, pattern: re.Pattern) -> pd.DataFrame:
    """Stack CSV blocks vertically in required mask/rate order."""
    blocks = []

    for mask in MASK_ORDER:
        for rate in RATE_ORDER:
            fname = None
            for f in os.listdir(folder):
                m = pattern.match(f)
                if not m:
                    continue
                if m.group("mask") == mask and int(m.group("rate")) == rate:
                    fname = f
                    break

            if fname is None:
                continue

            path = os.path.join(folder, fname)
            df = pd.read_csv(path)

            # block header
            header = pd.DataFrame(
                [[f"{mask} — {rate}% missing rate"] + [""] * (df.shape[1] - 1)],
                columns=df.columns,
            )
            blocks.append(header)
            blocks.append(df)
            blocks.append(pd.DataFrame([[""] * df.shape[1]], columns=df.columns))

    if not blocks:
        return pd.DataFrame()

    return pd.concat(blocks, ignore_index=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    app1_path = os.path.join(base_dir, APPENDIX1_DIR)
    app2_path = os.path.join(base_dir, APPENDIX2_DIR)

    if not os.path.isdir(app1_path):
        raise FileNotFoundError(f"Missing folder: {APPENDIX1_DIR}")
    if not os.path.isdir(app2_path):
        raise FileNotFoundError(f"Missing folder: {APPENDIX2_DIR}")

    df_app1 = _collect_blocks(app1_path, PAT_FIG5)
    df_app2 = _collect_blocks(app2_path, PAT_FIG6)

    out_path = os.path.join(base_dir, OUT_XLSX)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_app1.to_excel(writer, sheet_name="Appendix1", index=False)
        df_app2.to_excel(writer, sheet_name="Appendix2", index=False)

    print(f"Saved combined appendix tables to: {out_path}")


if __name__ == "__main__":
    main()
