# data.py
# Creates a leakage-safe dataset for the FastAPI backend/UI.
# Usage (Windows PowerShell):
#   py data.py --in "..\data\SL_food_prices_prepared (2).csv" --out "..\data\SL_food_prices_clean_noleak.csv"

import argparse
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def smart_read(path: str) -> pd.DataFrame:
    # Try flexible CSV parse
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
        if df.shape[1] == 1:  # wrong delimiter guessed
            for sep in [",", ";", "\t", "|"]:
                try:
                    df2 = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
                    if df2.shape[1] > 1:
                        return df2
                except Exception:
                    pass
        return df
    except Exception:
        pass
    # Excel fallback
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        pass
    # Other encodings
    for enc in ["latin1", "cp1252"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    raise RuntimeError(f"Could not read file: {path}")


def detect_date_column(df: pd.DataFrame) -> pd.Series:
    # common candidates
    for c in ["date", "Date", "period", "month_year", "obs_date", "time", "dt"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return s.dt.to_period("M").dt.start_time

    # yyyy-mm in text
    for c in df.columns:
        if df[c].dtype == object:
            smp = df[c].astype(str).head(20)
            if smp.str.contains(r"^\d{4}[-/]\d{1,2}$").any():
                s = pd.to_datetime(df[c].astype(str) + "-01", errors="coerce")
                if s.notna().any():
                    return s.dt.to_period("M").dt.start_time

    # year + month numeric
    lower = {x.lower(): x for x in df.columns}
    ycol = next((lower[k] for k in ["year", "yr", "yyyy"] if k in lower), None)
    mcol = next((lower[k] for k in ["month", "mn", "mm"] if k in lower), None)
    if ycol and mcol:
        y = pd.to_numeric(df[ycol], errors="coerce")
        m = pd.to_numeric(df[mcol], errors="coerce")
        s = pd.to_datetime({"year": y, "month": m, "day": 1}, errors="coerce")
        if s.notna().any():
            return s.dt.to_period("M").dt.start_time

    # yyyymm int
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        ok = vals.between(190001, 210012, inclusive="both")
        if ok.any():
            s = pd.to_datetime(vals.astype("Int64").astype(str) + "01", format="%Y%m%d", errors="coerce")
            if s.notna().any():
                return s.dt.to_period("M").dt.start_time

    return pd.Series(pd.NaT, index=df.index)


def build_commodity(df: pd.DataFrame) -> str:
    for c in ["commodity", "item", "product", "commodity_name"]:
        if c in df.columns:
            return c
    # derive from one-hots like commodity_Rice (imported)
    onehots = [c for c in df.columns if c.lower().startswith("commodity_")]
    if onehots:
        def get_name(row):
            cols = [c for c in onehots if pd.notna(row[c]) and float(row[c]) == 1.0]
            if not cols:
                return "Unknown"
            raw = cols[0].split("commodity_", 1)[-1]
            return raw.strip()
        df["commodity"] = df.apply(get_name, axis=1)
        return "commodity"
    df["commodity"] = "Commodity"
    return "commodity"


def build_market(df: pd.DataFrame) -> str:
    for c in ["market", "region", "district", "location", "city", "pop_region", "area", "market_name", "region_name"]:
        if c in df.columns:
            # coerce to string to avoid numeric codes being treated as numbers
            df[c] = df[c].astype(str)
            return c
    # derive from one-hots like region_Western Area / market_Freetown
    region_oh = [c for c in df.columns if c.lower().startswith(("region_", "market_", "district_"))]
    if region_oh:
        def get_region(row):
            cols = [c for c in region_oh if pd.notna(row[c]) and float(row[c]) == 1.0]
            if not cols:
                return "Unknown"
            raw = cols[0]
            for p in ["region_", "market_", "district_"]:
                if raw.lower().startswith(p):
                    return raw[len(p):]
            return raw
        df["market"] = df.apply(get_region, axis=1)
        df["market"] = df["market"].astype(str)
        return "market"
    df["market"] = "All-Markets"
    return "market"


def choose_price(df: pd.DataFrame) -> str:
    # prefer price_sll if present
    if "price_sll" in df.columns:
        return "price_sll"
    # otherwise any column with "price" but not lag/roll
    for c in df.columns:
        lc = c.lower()
        if "price" in lc and "_lag" not in lc and "_roll" not in lc:
            return c
    raise RuntimeError("No price column found (need a column containing 'price').")


def make_features_targets(df: pd.DataFrame, date_col: str, comm_col: str, mkt_col: str, price_col: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([comm_col, mkt_col, date_col])

    # Create features per (commodity, market)
    def add_group_feats(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col).copy()
        p = g[price_col].astype(float)
        g[f"{price_col}_lag1"] = p.shift(1)
        g[f"{price_col}_lag2"] = p.shift(2)
        g[f"{price_col}_lag3"] = p.shift(3)
        g[f"{price_col}_lag6"] = p.shift(6)
        g[f"{price_col}_roll3"] = p.rolling(3, min_periods=3).mean()
        g[f"{price_col}_roll6"] = p.rolling(6, min_periods=6).mean()
        # time-safe targets: future levels
        g["target_1m"] = p.shift(-1)
        g["target_3m"] = p.shift(-3)
        g["target_6m"] = p.shift(-6)
        return g

    df = df.groupby([comm_col, mkt_col], group_keys=False).apply(add_group_feats)

    # Drop rows that would cause leakage / NaNs (need all features + all targets)
    feat_cols = [c for c in df.columns if re.search(fr"^{re.escape(price_col)}_(lag|roll)", c)]
    target_cols = ["target_1m", "target_3m", "target_6m"]
    keep = df.dropna(subset=feat_cols + target_cols)
    # Keep only helpful columns
    out_cols = [date_col, comm_col, mkt_col, price_col] + feat_cols + target_cols
    return keep[out_cols].sort_values([comm_col, mkt_col, date_col]).reset_index(drop=True)


def main(in_path: str, out_path: str):
    df0 = smart_read(in_path)

    # date
    dt = detect_date_column(df0)
    if dt.isna().all():
        raise RuntimeError("Could not detect/construct a monthly date column.")
    df0["date"] = pd.to_datetime(dt).dt.to_period("M").dt.start_time

    # commodity & market
    comm_col = build_commodity(df0)
    mkt_col = build_market(df0)

    # price
    price_col = choose_price(df0)

    # numeric only for price
    df0[price_col] = pd.to_numeric(df0[price_col], errors="coerce")

    # build features & targets
    clean = make_features_targets(df0, "date", comm_col, mkt_col, price_col)

    # rename price column to a stable name the backend expects (optional)
    if price_col != "price_sll":
        clean = clean.rename(columns={price_col: "price_sll"})
        price_col = "price_sll"

    # save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False, encoding="utf-8")

    # simple report
    print("[OK] Saved:", out_path)
    print("Rows:", clean.shape[0], "Columns:", list(clean.columns)[:10], "...")
    print("Commodity examples:", clean["commodity"].drop_duplicates().head(10).tolist())
    print("Market examples:", clean["market"].drop_duplicates().head(10).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV/XLSX path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    args = ap.parse_args()
    main(args.in_path, args.out_path)
