# app.py
# ---------------------------------------------
# SMART Retail Price Forecasting API (FastAPI)
# - Uses SL_food_prices_prepared.csv ONLY
# - Region dropdown shows clean names (Eastern, North Western, Northern, Southern, Western Area)
# - "Current price" = last observation in 2024
# - 1m/3m/6m forecasts are produced for 2025 only (trained on data up to 2024)
# - Aggregation for region="All" (median by date)
# ---------------------------------------------
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# (Optional) Open CORS while testing; tighten later to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: ["https://<your-frontend>.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "service": "sl-backend", "docs": "/docs"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ---------------- Paths & dataset candidates ----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

#  Use ONLY this file (per your instruction)
DATA_CANDIDATES = [DATA_DIR / "SL_food_prices_prepared.csv"]

# ---------------- Globals ----------------
DF: Optional[pd.DataFrame] = None
DATE_COL: Optional[str] = None
PRICE_COL: Optional[str] = None
REGION_COL: Optional[str] = None            # resolved region/market column (textual)
TIDY_COMMODITY_COL: Optional[str] = None    # e.g., "commodity" in tidy data
WIDE_COMMODITY_MAP: Dict[str, str] = {}     # friendly name -> 'commodity_*' column (wide data)

# Canonical display order for Sierra Leone regions
CANON_REGIONS_ORDER = ["Eastern", "North Western", "Northern", "Southern", "Western Area"]
# Commodity labels we float to the top if present
CANON_COMMODITIES = ["Fish (bonga)", "Rice (imported)", "Oil (palm)"]


# ---------------- Utils ----------------
def _read_any(p: Path) -> pd.DataFrame:
    """Read CSV or Excel with minimal fuss."""
    return pd.read_excel(p) if p.suffix.lower() in (".xls", ".xlsx") else pd.read_csv(p, encoding="utf-8")


def _norm(s: Optional[str]) -> str:
    """Normalise text for case-insensitive matching."""
    return "" if s is None else str(s).strip().lower()


def _label_from_region_flag_col(colname: str) -> str:
    """
    Convert one-hot column like "region_north western" to "North Western".
    Used only if the dataset has one-hot region flags instead of a single region column.
    """
    label = colname[len("region_"):].strip().replace("_", " ")
    # Title case then fix known canonical forms
    label = " ".join(w.capitalize() for w in label.split())
    fixes = {
        "North Western": "North Western",
        "Western Area": "Western Area",
        "Eastern": "Eastern",
        "Northern": "Northern",
        "Southern": "Southern",
    }
    for k, v in fixes.items():
        if _norm(label) == _norm(k):
            return v
    return label


# ---------- NEW: helper to turn numeric code columns into names when a companion text column exists ----------
def _coerce_region_to_names(
    df: pd.DataFrame,
    region_col: str,
    low: List[str],
    cols: List[str],
) -> Tuple[pd.DataFrame, str]:
    """
    If region_col looks numeric (codes), replace it with a text label by
    mapping to a companion name column (e.g., pop_region_name, market_name).
    Returns (df, new_region_col).
    """
    def _mostly_numeric(s: pd.Series) -> bool:
        try:
            ss = s.dropna().astype(str).str.replace(r"\s+", "", regex=True)
            return (ss.str.fullmatch(r"\d+").mean() if len(ss) else 0.0) > 0.8
        except Exception:
            return False

    if not _mostly_numeric(df[region_col]):
        return df, region_col

    # ----- B) Companion-name mapping preference -----
    name_candidates = [
        "pop_region_name",              # prefer this
        "region_name", "region name",
        "market_name", "market name",
        "district_name", "district name",
        "area_name", "area name",
        "admin_name", "admin name",
        "province_name", "province name",
        "region",                       # sometimes this is already the human name
    ]
    name_col = None
    for k in name_candidates:
        if k in low:
            name_col = cols[low.index(k)]
            break

    if not name_col:
        # No companion name column found; stringify the codes and return
        df2 = df.copy()
        df2[region_col] = df2[region_col].astype(str).str.strip()
        return df2, region_col

    # Build a stable code->name map (use the mode name per code)
    df2 = df.copy()
    tmp = (
        df2[[region_col, name_col]]
        .dropna(subset=[region_col])
        .assign(
            _code=df2[region_col].astype(str).str.strip(),
            _name=df2[name_col].astype(str).str.strip(),
        )
        .groupby("_code")["_name"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty
             else (s.dropna().iloc[0] if not s.dropna().empty else np.nan))
    )
    df2["region_label"] = df2[region_col].astype(str).str.strip().map(tmp).fillna(
        df2[region_col].astype(str).str.strip()
    )
    return df2, "region_label"


def _detect_columns_and_prepare(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, str, str, Optional[str], Dict[str, str]]:
    """
    Detect key columns (date, price, region, commodity) and return a cleaned/typed DataFrame.

    IMPORTANT:
    - HARD-PREFER a textual region column; numeric market IDs are never exposed to the UI.
    """
    cols = [str(c).strip() for c in df.columns]
    low = [c.lower() for c in cols]

    # ---- DATE ----
    date_col = None
    for k in ["date", "month", "period", "obs_date"]:
        if k in low:
            date_col = cols[low.index(k)]
            break
    if not date_col:
        raise RuntimeError("Could not find a date/period column")

    # ---- PRICE ----
    price_col = None
    for k in ["price_sll", "retail_price_sll", "price_slll", "price"]:
        if k in low:
            price_col = cols[low.index(k)]
            break
    if not price_col:
        # fallback: any column containing "price"
        for i, c in enumerate(low):
            if "price" in c:
                price_col = cols[i]
                break
    if not price_col:
        raise RuntimeError("Could not find a price column")

    # ---------------------------
    # REGION COLUMN (force text)
    # ---------------------------

    # ----- A) Region candidates preference (names first; numeric last) -----
    region_col: Optional[str] = None
    region_candidates = [
        "pop_region_name",              # <— prefer this first
        "region",
        "region_name", "region name",
        "market_name", "market name",
        "district", "district_name", "district name",
        "province", "admin1", "admin_1", "admin_region",
        "pop_region", "pop region",    # often numeric; left here for detection
        "market",                       # numeric: keep LAST
    ]
    for k in region_candidates:
        if k in low:
            region_col = cols[low.index(k)]
            break

    # 1-hot fallback: synthesise from region_Eastern, region_Southern etc.
    df = df.copy()
    if region_col is None:
        region_flag_cols = [c for c in cols if c.lower().startswith("region_")]
        if region_flag_cols:
            def synth_region(row) -> Optional[str]:
                for c in region_flag_cols:
                    val = row.get(c)
                    try:
                        num = float(val); active = num > 0
                    except Exception:
                        active = str(val).strip().lower() in ("1", "true", "yes")
                    if active:
                        return _label_from_region_flag_col(c)
                return None
            df["region"] = df.apply(synth_region, axis=1)
            region_col = "region"

    if region_col is None:
        raise RuntimeError("Could not detect a region column")

    # ----- C) If picked region column is numeric, switch to a textual alternative or coerce codes -> names -----
    def _mostly_numeric(s: pd.Series) -> bool:
        try:
            ss = s.dropna().astype(str).str.replace(r"\s+", "", regex=True)
            return (ss.str.fullmatch(r"\d+").mean() if len(ss) else 0.0) > 0.8
        except Exception:
            return False

    if _mostly_numeric(df[region_col]):
        # Try textual alternatives first (prefer pop_region_name)
        for alt in [
            "pop_region_name",
            "region", "region_name", "region name",
            "market_name", "market name",
            "district", "district_name", "district name",
            "province", "admin1", "admin_1", "admin_region",
        ]:
            if alt in low:
                cand = cols[low.index(alt)]
                if not _mostly_numeric(df[cand]):
                    region_col = cand
                    break
        # Still numeric? Coerce using companion name column (B)
        if _mostly_numeric(df[region_col]):
            df, region_col = _coerce_region_to_names(df, region_col, low, cols)

    # ---- COMMODITY (tidy or wide) ----
    tidy_commodity_col = None
    for k in ["commodity", "item", "product"]:
        if k in low:
            tidy_commodity_col = cols[low.index(k)]
            break

    # Wide map: columns like "commodity_rice (imported)" => friendly label
    wide_map: Dict[str, str] = {}
    if tidy_commodity_col is None:
        for c in cols:
            lc = c.lower()
            if lc.startswith("commodity_"):
                raw = c[len("commodity_"):].strip()
                friendly = (
                    "Fish (bonga)" if _norm(raw) in ["fish (bonga)", "fish(bonga)", "bonga"] else
                    "Rice (imported)" if _norm(raw) in ["rice (imported)", "rice(imported)", "imported rice"] else
                    "Oil (palm)" if _norm(raw) in ["oil (palm)", "oil(palm)", "palm oil"] else
                    raw
                )
                wide_map[friendly] = c

    # ---- Clean & type ----
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col, region_col]).reset_index(drop=True)
    df[region_col] = df[region_col].astype(str).str.strip()

    return (
        df.sort_values(date_col).reset_index(drop=True),
        date_col,
        price_col,
        region_col,
        tidy_commodity_col,
        wide_map,
    )


def _load_dataset() -> None:
    """Try the single allowed file and validate."""
    global DF, DATE_COL, PRICE_COL, REGION_COL, TIDY_COMMODITY_COL, WIDE_COMMODITY_MAP
    last_err = None
    for p in DATA_CANDIDATES:
        if not p.exists():
            continue
        try:
            raw = _read_any(p)
            raw.columns = [str(c).strip() for c in raw.columns]
            df, date_col, price_col, region_col, tidy_col, wide_map = _detect_columns_and_prepare(raw)
            DF, DATE_COL, PRICE_COL, REGION_COL = df, date_col, price_col, region_col
            TIDY_COMMODITY_COL, WIDE_COMMODITY_MAP = tidy_col, dict(wide_map)
            print(
                f"[INFO] Loaded {len(DF)} rows. "
                f"date_col={DATE_COL} price_col={PRICE_COL} region_col={REGION_COL} "
                f"mode={'tidy' if TIDY_COMMODITY_COL else ('wide' if WIDE_COMMODITY_MAP else 'single')}"
            )
            return
        except Exception as e:
            last_err = e
            print(f"[WARN] Failed reading {p.name}: {e}")
    raise SystemExit(f"[FATAL] Could not read {DATA_CANDIDATES[0].name}. Last error: {last_err}")


def _available_commodities() -> List[str]:
    """Return ordered commodity list based on tidy or wide mode."""
    if TIDY_COMMODITY_COL:
        vals = DF[TIDY_COMMODITY_COL].dropna().astype(str).unique().tolist()  # type: ignore
        ordered = [c for c in CANON_COMMODITIES if c in vals]
        for v in vals:
            if v not in ordered:
                ordered.append(v)
        return ordered
    if WIDE_COMMODITY_MAP:
        ordered = [c for c in CANON_COMMODITIES if c in WIDE_COMMODITY_MAP]
        for v in WIDE_COMMODITY_MAP:
            if v not in ordered:
                ordered.append(v)
        return ordered
    # Single-series fallback
    return ["price"]


def _filter_by_selection(df: pd.DataFrame, commodity: str, region: str) -> pd.DataFrame:
    """
    Filter the dataset by commodity and region.
    - Tidy mode: filter rows where commodity == selected.
    - Wide mode: if commodity column exists, use it as a flag to subset rows.
    - Region: specific region filters rows; "All" is handled by aggregation later.
    """
    out = df
    # commodity
    if TIDY_COMMODITY_COL:
        if commodity and _norm(commodity) != "price":
            out = out[
                out[TIDY_COMMODITY_COL].astype(str).str.strip().str.lower() == _norm(commodity)  # type: ignore
            ]
    elif WIDE_COMMODITY_MAP and commodity and _norm(commodity) != "price":
        col = WIDE_COMMODITY_MAP.get(commodity)
        if col and col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            if s.notna().any():
                out = out[s.fillna(0) > 0]
            else:
                out = out[out[col].astype(str).str.lower().isin(["1", "true", "yes"])]
    # region (specific only; "All" is aggregated later)
    if region and _norm(region) not in ["", "all"]:
        out = out[out[REGION_COL].astype(str).str.strip().str.lower() == _norm(region)]
    return out


def _aggregate_if_all(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    If region=='All', aggregate by date (median price) so the series is single-valued.
    Otherwise just sort by date.
    """
    if _norm(region) in ["", "all"]:
        g = (
            df.groupby(DATE_COL, as_index=False)[PRICE_COL]
              .median()
              .sort_values(DATE_COL)
        )
        return g
    return df.sort_values(DATE_COL)


# --------------- Forecast helpers ---------------
def _holt_winters_forecast(y: pd.Series, periods: int) -> np.ndarray:
    """
    Try Holt-Winters with seasonality; fall back to a simple linear projection if not enough data.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        if len(y.dropna()) >= 18:
            fit = ExponentialSmoothing(
                y.astype(float),
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit(optimized=True)
            return fit.forecast(periods).astype(float).to_numpy()
    except Exception:
        # If statsmodels not available or model fails, use a simple linear drift.
        pass

    vals = y.dropna().astype(float).values
    if len(vals) == 0:
        return np.full(periods, np.nan)
    last = vals[-1]
    slope = (vals[-1] - vals[-7]) / 6.0 if len(vals) >= 7 else 0.0
    return np.array([last + slope * (i + 1) for i in range(periods)], dtype=float)


def _train_until_2024(df: pd.DataFrame) -> pd.DataFrame:
    """Keep data up to and including 2024-12-31 for model training and current-price computation."""
    return df[pd.to_datetime(df[DATE_COL]).dt.year <= 2024].copy()  # type: ignore


# --------------- API ----------------
app = FastAPI(title="SMART Retail Price Forecasting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)


@app.get("/health")
def health():
    """Quick status for debugging."""
    return {
        "ok": DF is not None,
        "rows": 0 if DF is None else len(DF),
        "date_col": DATE_COL,
        "price_col": PRICE_COL,
        "region_col": REGION_COL,
        #  region names (no numeric IDs)
        "regions_present": [] if DF is None else sorted(DF[REGION_COL].astype(str).unique()),
        "commodities_present": _available_commodities(),
        "mode": "tidy" if TIDY_COMMODITY_COL else ("wide" if WIDE_COMMODITY_MAP else "single"),
    }


@app.get("/options")
def options():
    """
    Return commodities only.
    Regions should be fetched via /regions?commodity=... so they reflect the chosen commodity.
    """
    return {"commodities": _available_commodities()}


@app.get("/regions", response_model=List[str])
def regions(commodity: str = Query(..., description="Commodity name")):
    """
    Return distinct regions for the selected commodity.
    "All" is included as the first option.
    """
    sub = _filter_by_selection(DF, commodity, region="All")  # type: ignore
    if sub is None or sub.empty:
        return ["All"]
    regs = sorted(sub[REGION_COL].dropna().astype(str).unique())
    # keep canonical ordering where possible
    ordered = [r for r in CANON_REGIONS_ORDER if r in regs]
    for r in regs:
        if r not in ordered:
            ordered.append(r)
    return ["All"] + ordered


@app.get("/series")
def series(commodity: str = Query("price"), region: str = Query("All"), months: int = Query(18)):
    """
    Return historical series for charts, optionally aggregated (median) when region='All'.
    """
    sub = _filter_by_selection(DF, commodity, region)  # type: ignore
    sub = _aggregate_if_all(sub, region)
    if months and months > 0:
        sub = sub.tail(months)
    pts = [{"date": pd.to_datetime(d).date().isoformat(), "y": float(v)}
           for d, v in zip(sub[DATE_COL], sub[PRICE_COL])]
    return {"points": pts}


@app.get("/metrics")
def metrics(commodity: str = Query(...), region: str = Query("All")):
    """
    Return:
      - current_price: the latest actual price in **2024**
      - fcst_1m / fcst_3m / fcst_6m: **2025** forecasts (Jan..Jun depending on data),
        produced by a model trained on data up to 2024-12-31.
    """
    # Filter, aggregate, and then clamp to 2024 for training/current price
    sub_all = _filter_by_selection(DF, commodity, region)  # type: ignore
    sub_all = _aggregate_if_all(sub_all, region)
    sub = _train_until_2024(sub_all)
    sub = sub.dropna(subset=[PRICE_COL, DATE_COL]).sort_values(DATE_COL)
    if sub.empty:
        return {
            "commodity": commodity, "region": region,
            "current_price": None, "as_of": None,
            "fcst_1m": None, "fcst_3m": None, "fcst_6m": None
        }

    # Current price as of last date in 2024
    current_price = float(sub[PRICE_COL].iloc[-1])
    last_date_2024 = pd.to_datetime(sub[DATE_COL].iloc[-1])

    # Train on 2024-and-earlier, forecast 6 months ahead
    fc6 = _holt_winters_forecast(sub[PRICE_COL], 6)

    # Future dates should live in 2025
    fdates = [(last_date_2024 + pd.DateOffset(months=i)).date().isoformat()
              for i in range(1, 7)]

    def safe(v):
        return float(v) if (v is not None and np.isfinite(v)) else None

    return {
        "commodity": commodity,
        "region": region,
        "current_price": current_price,
        "as_of": last_date_2024.date().isoformat(),  #  2024 date
        "fcst_1m": safe(fc6[0]) if len(fc6) >= 1 else None,  #  2025 month 1
        "fcst_3m": safe(fc6[2]) if len(fc6) >= 3 else None,  #  2025 month 3
        "fcst_6m": safe(fc6[5]) if len(fc6) >= 6 else None,  #  2025 month 6
        "future_dates": {"1m": fdates[0], "3m": fdates[2], "6m": fdates[5]},
    }


@app.get("/predict")
def predict(commodity: str, region: str = Query("All"), horizon: int = Query(1)):
    """
    Legacy endpoint:
    - current_price = last 2024 observation
    - pred_1m/3m/6m are 2025 forecasts
    """
    sub_all = _filter_by_selection(DF, commodity, region)  # type: ignore
    sub_all = _aggregate_if_all(sub_all, region)
    sub = _train_until_2024(sub_all)
    sub = sub.dropna(subset=[PRICE_COL, DATE_COL]).sort_values(DATE_COL)
    if sub.empty:
        return {"error": "no data for selection"}, 404

    current_price = float(sub[PRICE_COL].iloc[-1])
    last_date_2024 = pd.to_datetime(sub[DATE_COL].iloc[-1])
    fc6 = _holt_winters_forecast(sub[PRICE_COL], 6)
    fdates = [(last_date_2024 + pd.DateOffset(months=i)).date().isoformat() for i in range(1, 7)]

    def pct(v):
        return None if (v is None or not np.isfinite(v) or abs(current_price) < 1e-9) else (float(v) - current_price) / current_price * 100.0

    bundle = {
        "pred_1m": float(fc6[0]) if np.isfinite(fc6[0]) else None,
        "pred_3m": float(fc6[2]) if np.isfinite(fc6[2]) else None,
        "pred_6m": float(fc6[5]) if np.isfinite(fc6[5]) else None,
        "pct_change_1m": pct(fc6[0]),
        "pct_change_3m": pct(fc6[2]),
        "pct_change_6m": pct(fc6[5]),
        "future_dates": {"1m": fdates[0], "3m": fdates[2], "6m": fdates[5]},  #  2025
        "future_path": [{"date": fdates[i], "forecast": float(fc6[i]) if np.isfinite(fc6[i]) else None} for i in range(6)],
    }

    snap = {1: ("pred_1m", "1m"), 3: ("pred_3m", "3m"), 6: ("pred_6m", "6m")}.get(horizon, ("pred_1m", "1m"))
    key, tag = snap
    return {
        "commodity": commodity,
        "region": region,
        "horizon": horizon,
        "forecast_price": bundle[key],
        "current_price": current_price,                #  2024
        "pct_change": bundle[f"pct_change_{tag}"],
        "future_date": bundle["future_dates"][tag],    # 2025 target month
        "kpi": bundle,
    }


# Load data at startup
_load_dataset()

# To run locally:
# uvicorn app:app --reload --port 8000
