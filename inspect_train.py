# backend/inspect_train.py
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent / "../data"
FILES = {
    1: DATA / "Region_with_lags_POSTONLY_train_1M.csv",
    3: DATA / "Region_with_lags_POSTONLY_train_3M.csv",
}

TARGETS = [
    "price_sll",
    "commodity_oil (palm)",
    "commodity_fish (bonga)",
    "commodity_rice (imported)",
]

def cols_for(h, df):
    return {t: f"{t}_lead{h}" for t in TARGETS if f"{t}_lead{h}" in df.columns}

for h, p in FILES.items():
    if not p.exists():
        print(f"[MISSING] {h}M file: {p}")
        continue
    df = pd.read_csv(p)
    if "region" not in df.columns:
        print(f"[WARN] {p.name} has no 'region' column.")
        continue
    present = cols_for(h, df)
    print(f"\n=== {h}M: {p.name} ===")
    print("Targets found:", list(present.values()))
    for reg, g in df.groupby("region", dropna=False):
        ok = [c for c in present.values() if c in g.columns and g[c].notna().sum() >= 8]
        print(f"  - {reg}: {len(g)} rows, targets with >=8 non-NA rows: {ok}")
