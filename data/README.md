# data/

Git ignores everything in this folder except this file. Do not commit weather data, exports or built tables.

## Source 1: NASA POWER hourly point data (public)

- Provider: NASA Langley Research Center, POWER project, <https://power.larc.nasa.gov>
- Terms: public data. Cite the POWER project as the data source.
- Download with the CLI (UTC, cached in `cache/`):
  `skyfusion fetch-power --start 20010101 --end 20241231`
  The request uses `community=RE` and the parameters `T2M, RH2M, WS2M, PS, PRECTOTCORR,
  ALLSKY_SFC_SW_DWN, ALLSKY_SFC_LW_DWN` at `SKYFUSION_LAT`, `SKYFUSION_LON`.
- Files from the POWER Data Access Viewer are also accepted. They can be in local standard time
  (`... in LST` in the header). The loader shifts LST to UTC with `round(lon / 15)` hours
  (−6 h at Dallas/Fort Worth, lon −97.04).
- Missing values are `-999`. The loader changes them to NaN.

| Column | Meaning | Unit |
|---|---|---|
| `YEAR, MO, DY, HR` | Time (LST or UTC, see the header) | |
| `T2M` | Temperature at 2 m | °C |
| `RH2M` | Relative humidity at 2 m | % |
| `WS2M` | Wind speed at 2 m | m/s |
| `PS` | Surface pressure | kPa |
| `PRECTOTCORR` | Precipitation | mm/h |
| `ALLSKY_SFC_SW_DWN` | Downward shortwave irradiance | W/m² |
| `ALLSKY_SFC_LW_DWN` | Downward longwave irradiance | W/m² |

## Source 2: station observations (OpenWeather History Bulk, licensed)

- Provider: OpenWeather, "History Bulk" export for one location, <https://openweathermap.org/history-bulk>
- Terms: a paid product with its own licence. Do not redistribute the file.
- Save it as `data/station_openweather.csv`.
- Used columns: `dt` (Unix seconds, UTC) or `dt_iso`, `temp`, `dew_point`, `feels_like` (kelvin),
  `pressure` (sea-level hPa), `grnd_level` (surface hPa, often empty), `humidity` (%),
  `wind_speed` (m/s at 10 m), `rain_1h` (mm, empty when dry), `clouds_all` (%).

## Build the table

```bash
skyfusion build-dataset --station data/station_openweather.csv --power cache/power_*.csv --out data/fused.csv
```

## Synthetic data (offline demo and tests)

`skyfusion synth --out data/synthetic` writes `station_openweather.csv` (UTC, kelvin, hPa) and
`power_lst.csv` (LST header, -999 fill values for the last 60 days of irradiance) from one hidden
synthetic truth for 2018 to 2022. No real observation is in these files.
