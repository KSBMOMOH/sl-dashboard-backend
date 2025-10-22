# clean_data.py
# Leakage-safe preparation for SL food prices:
# - Builds CPI-deflated price (if CPI present)
# - Creates lags (t-1, t-3), rolling mean (past-only), seasonal dummies, and spatial one-hots
# - Generates future targets (1m/3m/6m) without leakage
# - Saves a companion feature list for inference alignment
#
# Usage (PowerShell, from backend folder):
#   py .\clean_data.py --in "..\data\SL_food_prices_prepared.csv" --out "..\data\SL_food_prices_clean_noleak.csv" --cpi-col CPI

import argparse
import re
from pathlib import Path
import json
import numpy as np
import pandas as pd

# --------- IO helpers ---------
def smart_read(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
        if df.shape[1] == 1:
            for sep in [",",";","\t","|"]:
                try:
                    df2 = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
                    if df2.shape[1] > 1:
                        return df2
                except Exception:
                    pass
        return df
    except Exception:
        pass
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        pass
    for enc in ["latin1", "cp1252"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    raise RuntimeError(f"Could not read file: {path}")

def detect_date_column(df: pd.DataFrame) -> pd.Series:
    for c in ["date","Date","period","month_year","obs_date","time","dt"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return s.dt.to_period("M").dt.start_time
    for c in df.columns:
        if df[c].dtype == object:
            smp = df[c].astype(str).head(20)
            if smp.str.contains(r"^\d{4}[-/]\d{1,2}$").any():
                s = pd.to_datetime(df[c].astype(str) + "-01", errors="coerce")
                if s.notna().any():
                    return s.dt.to_period("M").dt.start_time
    lower = {x.lower(): x for x in df.columns}
    ycol = next((lower[k] for k in ["year","yr","yyyy"] if k in lower), None)
    mcol = next((lower[k] for k in ["month","mn","mm"] if k in lower), None)
    if ycol and mcol:
        s = pd.to_datetime({"year": pd.to_numeric(df[ycol], errors="coerce"),
                            "month": pd.to_numeric(df[mcol], errors="coerce"),
                            "day": 1}, errors="coerce")
        if s.notna().any():
            return s.dt.to_period("M").dt.start_time
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any() and vals.between(190001,210012, inclusive="both").any():
            s = pd.to_datetime(vals.astype("Int64").astype(str) + "01", format="%Y%m%d", errors="coerce")
            if s.notna().any():
                return s.dt.to_period("M").dt.start_time
    return pd.Series(pd.NaT, index=df.index)

def build_commodity(df: pd.DataFrame) -> str:
    for c in ["commodity","item","product","commodity_name"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
            return c
    onehots = [c for c in df.columns if c.lower().startswith("commodity_")]
    if onehots:
        def get_name(row):
            cols = [c for c in onehots if pd.notna(row[c]) and float(row[c]) == 1.0]
            return (cols[0].split("commodity_",1)[-1].strip() if cols else "Unknown")
        df["commodity"] = df.apply(get_name, axis=1).astype(str)
        return "commodity"
    df["commodity"] = "Commodity"
    return "commodity"

def build_market(df: pd.DataFrame) -> str:
    for c in ["market","region","district","location","city","pop_region","area","market_name","region_name"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
            return c
    region_oh = [c for c in df.columns if c.lower().startswith(("region_","market_","district_"))]
    if region_oh:
        def get_region(row):
            cols = [c for c in region_oh if pd.notna(row[c]) and float(row[c]) == 1.0]
            raw = cols[0] if cols else "Unknown"
            for pref in ["region_","market_","district_"]:
                if raw.lower().startswith(pref):
                    return raw[len(pref):]
            return raw
        df["market"] = df.apply(get_region, axis=1).astype(str)
        return "market"
    df["market"] = "All-Markets"
    return "market"

def choose_price(df: pd.DataFrame) -> str:
    if "price_sll" in df.columns:
        return "price_sll"
    for c in df.columns:
        lc = c.lower()
        if "price" in lc and "_lag" not in lc and "_roll" not in lc:
            return c
    raise RuntimeError("No price column found (need a column containing 'price').")

# --------- CPI deflation ---------
def make_real_price(df: pd.DataFrame, price_col: str, cpi_col: str | None) -> pd.Series:
    p = pd.to_numeric(df[price_col], errors="coerce")
    if cpi_col is None or cpi_col not in df.columns:
        return p
    cpi = pd.to_numeric(df[cpi_col], errors="coerce")
    med = np.nanmedian(cpi)
    if np.isfinite(med) and med > 10:
        # CPI ~ base=100
        return p / (cpi / 100.0)
    else:
        # CPI ~ chain-linked (around 1-3)
        return p / cpi

# --------- Group-safe engineering ---------
def winsorize_changes(g: pd.DataFrame, price_series: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
    """Conservative winsorization on MoM % changes, using only past info."""
    p = pd.to_numeric(price_series, errors="coerce")
    chg = p.pct_change()  # uses t-1 -> no future info
    lo, hi = chg.quantile(limits[0]), chg.quantile(limits[1])
    chg_clip = chg.clip(lower=lo, upper=hi)
    base = p.iloc[0]
    path = (1 + chg_clip).fillna(1.0).cumprod() * base
    return path  # smoothed path (still based only on past info up to t)

def make_features_targets(df: pd.DataFrame, price_col: str, add_spatial=True, add_seasonal=True) -> pd.DataFrame:
    df = df.sort_values(["commodity","market","date"]).copy()

    def per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        p = pd.to_numeric(g[price_col], errors="coerce")
        ps = winsorize_changes(g, p)

        # Required lags and rolling (PAST ONLY)
        g["lag1"] = p.shift(1)
        g["lag3"] = p.shift(3)
        # rolling mean over previous 3 months; exclude current by shifting before rolling
        g["roll3_mean"] = p.shift(1).rolling(3, min_periods=3).mean()

        # Future level targets (for model training files)
        g["target_1m"] = p.shift(-1)
        g["target_3m"] = p.shift(-3)
        g["target_6m"] = p.shift(-6)

        # Seasonals
        g["month"] = g["date"].dt.month.astype(int)
        if add_seasonal:
            m_d = pd.get_dummies(g["month"], prefix="m", drop_first=True)
            g = pd.concat([g, m_d], axis=1)

        return g

    out = df.groupby(["commodity","market"], group_keys=False).apply(per_group)

    # Spatial one-hots
    if add_spatial:
        reg_d = pd.get_dummies(out["market"].astype("category"), prefix="region", drop_first=True)
        out = pd.concat([out, reg_d], axis=1)

    # Keep only rows with full past for features (avoid leakage)
    needed_feats = ["lag1","lag3","roll3_mean"]
    out = out.dropna(subset=needed_feats).copy()

    # Select final columns (targets kept for training convenience)
    base_cols = ["date","commodity","market", price_col, "lag1","lag3","roll3_mean","month"]
    month_d_cols = [c for c in out.columns if re.fullmatch(r"m_\d+", c)]
    region_cols = [c for c in out.columns if c.startswith("region_")]
    target_cols = ["target_1m","target_3m","target_6m"]
    keep = base_cols + month_d_cols + region_cols + target_cols

    return out[keep].sort_values(["commodity","market","date"]).reset_index(drop=True)

def main(in_path: str, out_path: str, cpi_col: str | None):
    raw = smart_read(in_path)

    # 1) date
    dt = detect_date_column(raw)
    if dt.isna().all():
        raise RuntimeError("Could not detect/construct a monthly date column.")
    raw["date"] = pd.to_datetime(dt).dt.to_period("M").dt.start_time

    # 2) commodity & market
    comm_col = build_commodity(raw)
    mkt_col  = build_market(raw)
    raw = raw.rename(columns={comm_col: "commodity", mkt_col: "market"})
    raw["commodity"] = raw["commodity"].astype(str).str.strip()
    raw["market"]    = raw["market"].astype(str).str.strip()

    # 3) price (nominal), then real price if CPI present
    price_nom = choose_price(raw)
    raw[price_nom] = pd.to_numeric(raw[price_nom], errors="coerce")
    raw["price_real"] = make_real_price(raw, price_nom, cpi_col)

    clean = make_features_targets(raw[["date","commodity","market","price_real"]].copy(), "price_real",
                                  add_spatial=True, add_seasonal=True)

    # Save output
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_p, index=False, encoding="utf-8")

    # Save the feature list for inference alignment (drop identifiers & targets)
    feature_cols = [c for c in clean.columns
                    if c not in {"date","commodity","market","price_real","target_1m","target_3m","target_6m"}]
    feat_path = out_p.with_suffix(".features.json")
    feat_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    # Small report
    print("[OK] Saved:", out_p)
    print("Rows:", clean.shape[0])
    print("Columns:", list(clean.columns))
    print("Features saved to:", feat_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV/XLSX path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    ap.add_argument("--cpi-col", dest="cpi_col", default=None, help="CPI column name (optional)")
    args = ap.parse_args()
    main(args.in_path, args.out_path, args.cpi_col)
