# Backend
Place your `SL_food_prices_cleaned2.csv` inside `../data/` relative to `app.py`.
Create a venv and install requirements:

```
py -3.11 -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
python app.py
```
# Rewritten Backend (FastAPI)

- Trains per (commodity, market) models on load using leakage-safe features from the cleaned CSV.
- Endpoints:
  - `GET /api/options`
  - `GET /api/metrics`
  - `GET /api/forecast?commodity=...&market=...`
  - `GET /api/history?commodity=...&market=...&months=18`

## Run

```bash
# In sl-forecast-dashboard-vscode\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# point to your cleaned CSV (e.g., ..\data\SL_food_prices_clean_noleak.csv)
$env:DATA_PATH = (Resolve-Path ..\data\SL_food_prices_clean_noleak.csv).Path

python .\app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
