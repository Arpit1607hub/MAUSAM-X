# MAUSAM-X prototype (SIH26081)

Adaptive blending of real NWP and AI forecasts (NOAA GFS, ECMWF IFS, ECMWF AIFS)
that learns each model's skill per location, weather regime, lead day and
atmospheric stability, and outputs calibrated probabilities.

## Run
    pip install -r requirements.txt
    uvicorn main:app --reload
Dashboard: http://127.0.0.1:8000   API docs: http://127.0.0.1:8000/docs
Needs internet for live data (Open-Meteo, no API key). Scenario mode and the
demo replay work offline.

## Two views
- **Public view** (default, mobile-first, English/Hindi): 3-day outlook for any
  place in India (search or GPS). One rule (public.py) decides colour, headline
  and chance together, so they never contradict. Uncertainty is shown as
  "3 in 10" with dots.
- **Expert view** (`/#expert`): the full fusion console for officials,
  forecasters and judges.

## Admin actions
Recording observations, learning from history, demo replay and reset change
the skill record. Without `MAUSAMX_ADMIN_KEY` they only work from the server
computer. For any public deployment set a passcode:
    MAUSAMX_ADMIN_KEY=your-passcode uvicorn main:app

## First real demo
1. Pick a location, keep "Rain (auto)".
2. "Learn from the last 60 days": replays archived GFS/IFS/AIFS forecasts at
   1-3 day leads against ERA5, oldest first (no look-ahead). About 180 cases.
3. "Get latest forecasts": blends today's real forecasts with that skill.

## Test
    pytest -q      # 35 tests; the data pipeline is tested against mock
                   # responses in Open-Meteo's JSON format

## Files
- engine.py      skill engine: contexts with back-off, bias correction,
                 inverse-MSE weights, extreme flag, probabilities, Brier score
- datasource.py  Open-Meteo live forecasts, Previous Runs API, ERA5 archive
- public.py      one rule turning engine output into the public message
- main.py        FastAPI server; serves index.html
- index.html     dashboard (no external scripts; Google Fonts optional)

## Known limits (say these before judges ask)
- Truth is ERA5 reanalysis, not rain gauges. Swap in IMD gridded rainfall
  (e.g. the imdlib package) inside datasource.fetch_observations.
- Point forecasts at city coordinates, not a grid.
- AIFS is 6-hourly in open data, so its daily max temperature is biased low;
  bias correction absorbs part of this.
- NCUM (NCMRWF) is not openly available through this API; add it to
  MODEL_INFO in engine.py and MODEL_IDS in datasource.py when you have access.
- Demo replay uses synthetic data.

Data: Open-Meteo.com (CC BY 4.0); NOAA GFS; ECMWF IFS/AIFS open data (CC BY 4.0).
