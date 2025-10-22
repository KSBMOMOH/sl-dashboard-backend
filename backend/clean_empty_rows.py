import sys
import pandas as pd
from pathlib import Path

# Usage:
#   py clean_empty_rows.py <input_csv> [output_csv]
# Example:
#   py clean_empty_rows.py ..\data\Food_Price_Cleaned_IQR_filled.csv ..\data\Food_Price_Cleaned_noempty.csv

def main():
    if len(sys.argv) < 2:
        print("Usage: py clean_empty_rows.py <input_csv> [output_csv]")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_name(inp.stem + "_noempty.csv")

    # Load with robust encodings
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(inp, encoding=enc)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"Could not read {inp} ({last_err})")

    original = len(df)

    # Normalize common text columns (trim spaces) if present
    for col in ["region", "market", "commodity"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}, inplace=True)

    # Drop rows that are completely empty
    df = df.dropna(how="all")

    # Define key columns that must NOT be empty if they exist in the file
    key_cols = [c for c in ["date", "region", "commodity", "price_sll"] if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols)

    # If date exists, coerce bad dates and drop them
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    # Optional: drop duplicate rows
    df = df.drop_duplicates()

    removed = original - len(df)
    print(f"[OK] Loaded {original} rows; removed {removed} empty/invalid; saving {len(df)} rows -> {out}")

    # Save
    df.to_csv(out, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
