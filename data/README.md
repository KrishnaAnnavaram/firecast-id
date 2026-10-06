# Data

Git does not track the files in this folder, only this README. No detection file, series or climate file is in
the repository.

## Synthetic data (default, no download)

```bash
firecast synth --out-dir data/synthetic --start 2008-01-01 --end 2023-12-31
firecast synth --out-dir data/synthetic --detections-from 2022   # also detection rows in the FIRMS layout
```

The generator (`src/firecast_id/synthetic.py`) writes `daily_series.csv`, `climate.csv` and, with
`--detections-from`, `raw/synthetic_<year>.csv`. The tests and `firecast demo` use it.

## NASA FIRMS active-fire detections

| Item | Value |
|---|---|
| Source | NASA FIRMS (Fire Information for Resource Management System), MODIS Collection 6.1 active-fire detections |
| URL | https://firms.modaps.eosdis.nasa.gov/country/ (yearly country files) and https://firms.modaps.eosdis.nasa.gov/api/ (API) |
| License and terms | NASA data, free to use. Cite FIRMS as the FIRMS website asks. The API needs a free MAP_KEY. |
| Expected files | One CSV per year in `data/modis_indonesia/` (for example `modis_2019_Indonesia.csv`), or `firecast fetch` output |
| Size | About 110 MB for 2000-2023 (about 1.5 million detections) |

Columns that `firecast build-series` reads:

| Column | Required | Meaning |
|---|---|---|
| `acq_date` | Yes | Acquisition date (`YYYY-MM-DD`) |
| `confidence` | Yes | MODIS: 0..100. VIIRS: `l`, `n`, `h` (read as 10, 50, 90) |
| `type` | Yes | 0 = presumed vegetation fire, 1 = active volcano, 2 = other static land source, 3 = offshore |
| `frp` | No | Fire radiative power (MW). The daily sum becomes the `frp_sum` column |
| `latitude`, `longitude`, `acq_time`, `satellite`, `instrument`, `daynight` | No | Read but not used |

Steps:

1. Download the yearly Indonesia MODIS files from the FIRMS country page into `data/modis_indonesia/`.
   Or set `FIRMS_MAP_KEY` and run `firecast fetch --start 2023-01-01 --end 2023-12-31`.
2. Run `firecast build-series`. It keeps `type == 0` and `confidence >= 30` (change with `FIRECAST_FIRE_TYPES` and
   `FIRECAST_MIN_CONFIDENCE`) and writes `data/daily_series.csv` with a complete calendar. Days with no detection have 0.

## Climate drivers (optional)

| Item | Value |
|---|---|
| Example sources | NOAA Oceanic Nino Index (ONI), https://www.cpc.ncep.noaa.gov/ ; Dipole Mode Index (DMI) from NOAA PSL, https://psl.noaa.gov/ ; CHIRPS monthly rainfall, https://www.chc.ucsb.edu/data/chirps |
| License and terms | Public data from the agencies. Check the terms of each source. |
| Expected file | A CSV with a `month` column (`YYYY-MM`) and one numeric column per driver, for example `month,oni,dmi,rain_mm` |
| Use | Set `FIRECAST_CLIMATE` or pass `--climate`. A monthly value is used only from month end + `FIRECAST_CLIMATE_LAG_DAYS` (default 30). |
