<div align="center">

# skyfusion — Station and NASA POWER Fusion for Hourly Temperature Forecasts

**skyfusion is a forecasting pipeline for weather analysts who combine a ground station with NASA POWER data. It takes the two raw sources through these steps to scored forecasts for the next 1 to 24 hours:**

`load in UTC` → `check and fuse like-for-like` → `direct multi-horizon samples` → `baselines and models` → `skill and ablation`.

![Horizons](https://img.shields.io/badge/Horizons-1_to_24_h-1F3864?style=for-the-badge)
![Models](https://img.shields.io/badge/Models-6-2E5FD9?style=for-the-badge)
![CLI commands](https://img.shields.io/badge/CLI_commands-6-6E86E8?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-25_passing-3DA35B?style=for-the-badge)
![Offline demo](https://img.shields.io/badge/Offline_demo-Yes-F5C542?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-A0399B?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-UTC_grid-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ridge_%2B_boosting-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-optional_LSTM-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NASA POWER](https://img.shields.io/badge/NASA_POWER-API_client-0B3D91?style=flat-square&logo=nasa&logoColor=white)
![Docs](https://img.shields.io/badge/Docs-ASD--STE100-5D6D7E?style=flat-square)

**[Summary](#1-summary)** ·
**[Workflow](#4-the-end-to-end-workflow)** ·
**[Run it](#10-how-to-run-skyfusion)** ·
**[Configuration](#104-environment-variables)** ·
**[Known problems](#13-known-problems)** ·
**[Glossary](#15-glossary)**

</div>

> [!NOTE]
> This README uses ASD-STE100 Simplified Technical English. The writing rules and the project
> vocabulary are in [`docs/ste-style-guide.md`](docs/ste-style-guide.md). Each term in the
> [Glossary](#15-glossary) has only one meaning.

---

skyfusion forecasts the observed station temperature for each of the next 24 hours.
It puts the station data and the NASA POWER data on one UTC grid.
It averages two columns only when they measure the same quantity.
One model call gives all 24 horizons, so no prediction goes back into the inputs.
Each model is compared with persistence, and the station-only and fused feature sets are compared on identical samples.

This README is the **one location that explains all of skyfusion**. It gives these topics:

- the general design
- each component and its procedure, step by step
- the decision rules
- the data map
- the runbook
- the validation results and the known problems

| If you are… | Read |
|---|---|
| A manager or reviewer | [1](#1-summary), [3](#3-design-rules), [4](#4-the-end-to-end-workflow), [12](#12-validation-results), [14](#14-key-points) |
| A developer who joins the project | All sections, in sequence. Keep [10](#10-how-to-run-skyfusion) and [13](#13-known-problems) open while you work |
| An operator who runs skyfusion | [10](#10-how-to-run-skyfusion), then the section for the component that you use |

---

## Table of contents

1. 🧭 [Summary](#1-summary)
2. 🏗️ [How skyfusion is built](#2-how-skyfusion-is-built)
   - 2.1 [Components](#21-components)
   - 2.2 [System context](#22-system-context)
   - 2.3 [Repository layout](#23-repository-layout)
3. 🛡️ [Design rules](#3-design-rules)
4. 🔄 [The end-to-end workflow](#4-the-end-to-end-workflow)
   - 4.1 [Full flow](#41-full-flow)
   - 4.2 [The life cycle of one forecast](#42-the-life-cycle-of-one-forecast)
   - 4.3 [Who does which step](#43-who-does-which-step)
5. 🔵 [Ingestion and UTC alignment](#5-ingestion-and-utc-alignment)
6. 🟢 [Quality control and fusion](#6-quality-control-and-fusion)
7. 🟣 [Samples, split and models](#7-samples-split-and-models)
8. ⚖️ [The decision rules](#8-the-decision-rules)
9. 🗂️ [Data and file map](#9-data-and-file-map)
10. ▶️ [How to run skyfusion](#10-how-to-run-skyfusion)
    - 10.1 [Prerequisites](#101-prerequisites) · 10.2 [Installation](#102-installation) · 10.3 [Run skyfusion](#103-run-skyfusion) · 10.4 [Environment variables](#104-environment-variables)
11. 🧩 [How to extend skyfusion](#11-how-to-extend-skyfusion)
12. ✅ [Validation results](#12-validation-results)
13. ⚠️ [Known problems](#13-known-problems)
14. 📌 [Key points](#14-key-points)
15. 📖 [Glossary](#15-glossary)
16. 📄 [License](#16-license)

---

## 1. Summary

**The problem.** A ground station and NASA POWER describe the same site, but they use different times, units and quantities. These questions are difficult:

- How do you align a UTC station file and a POWER file in local standard time?
- Which columns can you average, and which columns only look similar?
- What do you do with the POWER fill value `-999`?
- How do you forecast many hours ahead without a recursive loop that feeds errors back?
- Does the second source improve the forecast, compared with persistence and on the same target?

skyfusion gives each of these questions its own component. Each component has unit tests.

| Item | Value |
|---|---|
| Input | OpenWeather History Bulk CSV (station) and NASA POWER hourly CSV or API data |
| Output | A UTC hourly table, MAE, RMSE and skill for each horizon, an ablation with intervals, a 24-hour forecast |
| Components | **9** modules: config, ingest, fuse, synthetic, windows, models, lstm, evaluate, cli |
| Providers | NASA POWER API (optional download), PyTorch (optional LSTM) |
| Offline mode | Synthetic files in both raw formats, all loaders, all baselines, ridge and boosting |
| Safety | One UTC index. The target is never averaged and never filled. No sample crosses a split boundary |
| Tests | **25** pass in CI (`.[dev]` only) and 1 skips (`lstm` extra). With the `lstm` extra, all 26 pass |

```mermaid
flowchart LR
    ST["Station (UTC)"] --> A["UTC grid"]
    PW["POWER (LST, -999)"] --> A
    A --> B["QC + fusion rules"] --> C["Direct 24 h samples"] --> D["Models"] --> OUT["Skill and ablation"]
```

---

## 2. How skyfusion is built

### 2.1 Components

| Component | Module | Purpose |
|---|---|---|
| Settings | `src/skyfusion/config.py` | Site, split years, window, horizon, seed |
| Ingestion | `src/skyfusion/ingest.py` | POWER parser (LST to UTC, -999 to NaN), POWER API client with cache, OpenWeather parser |
| Fusion | `src/skyfusion/fuse.py` | UTC grid, quality control, fusion rules, forward fill, mask columns |
| Synthetic data | `src/skyfusion/synthetic.py` | One hidden truth written in both raw file formats |
| Samples | `src/skyfusion/windows.py` | Direct multi-horizon samples, split by year with an embargo |
| Models | `src/skyfusion/models.py` | Persistence, seasonal naive, climatology, ridge, gradient boosting |
| LSTM | `src/skyfusion/lstm.py` | Optional direct multi-horizon LSTM with early stopping |
| Evaluation | `src/skyfusion/evaluate.py` | MAE, RMSE, skill, ablation with a day-block bootstrap |
| CLI | `src/skyfusion/cli.py` | The `skyfusion` command with 6 subcommands |

The component map shows which module calls which module. An arrow points from the caller to the module that it uses.

```mermaid
flowchart TB
    CLI["cli.py<br/>skyfusion command"]
    CFG["config.py<br/>Settings"]
    subgraph DATA["Data in"]
        SYN["synthetic.py<br/>write_synthetic"]
        ING["ingest.py<br/>read_power_csv, fetch_power,<br/>read_openweather_csv"]
        FUS["fuse.py<br/>build_dataset, feature_sets"]
    end
    subgraph LEARN["Samples and models"]
        WIN["windows.py<br/>make_samples, split_by_year"]
        MOD["models.py<br/>baselines, RidgeDirect, GBMDirect"]
        LST["lstm.py<br/>LSTMDirect, extra lstm"]
        EVA["evaluate.py<br/>run_experiment"]
    end

    CLI --> CFG
    CLI --> SYN
    CLI --> ING
    CLI --> FUS
    CLI --> EVA
    CLI -- "forecast" --> WIN
    CLI -- "forecast" --> MOD
    EVA --> FUS
    EVA --> WIN
    EVA --> MOD
    EVA -- "--lstm" --> LST
    LST --> MOD
    MOD --> WIN
```

### 2.2 System context

```mermaid
flowchart TB
    U["Analyst"] --> APP["skyfusion CLI"]
    API["NASA POWER API (optional)"] --> APP
    OW["OpenWeather export (licensed file)"] --> APP
    APP --> C["cache/ (POWER downloads)"]
    APP --> D["data/ (fused table)"]
    APP --> T["PyTorch (optional)"]
```

### 2.3 Repository layout

```
skyfusion/
├── .github/workflows/ci.yml   # CI: Python 3.11, pip install -e ".[dev]", pytest -q
├── data/README.md             # sources, terms, columns (data files are git-ignored)
├── docs/ste-style-guide.md    # writing rules and project vocabulary
├── src/skyfusion/             # the 9 modules in 2.1
├── tests/                     # 26 unit tests (1 needs the lstm extra), synthetic data only
├── .env.example               # variable names only
└── pyproject.toml             # core deps: numpy, pandas, scikit-learn, pydantic. Extras: lstm, dev
```

---

## 3. Design rules

### 3.1 One time standard: UTC
Each loader returns a timezone-aware UTC index. An LST POWER file is shifted by `round(lon / 15)` hours (−6 h at lon −97.04). `build_dataset` rejects a table without a UTC index. A test checks that the two temperatures agree best at lag 0, not at lag 6.

```mermaid
flowchart LR
    PF[/"POWER file<br/>YEAR, MO, DY, HR"/] --> STD{"Header says<br/>in LST or in UTC?"}
    STD -- "LST" --> SH["UTC = local time minus<br/>round of lon / 15 hours.<br/>lon −97.04: plus 6 h"]
    STD -- "UTC" --> NO["No shift"]
    STD -- "neither" --> ERR[/"IngestError"/]
    SH --> UTC[("One UTC hourly grid")]
    NO --> UTC
    OW[/"OpenWeather file<br/>dt in Unix seconds"/] --> UTC
    UTC --> CHK{"build_dataset:<br/>each index is UTC?"}
    CHK -- "no" --> ERR2[/"ValueError"/]
```

### 3.2 Fuse only the same quantity
Relative humidity at 2 m, precipitation and surface pressure can be averaged. Sea-level pressure and surface pressure are different, and wind at 10 m and wind at 2 m are different. These columns stay separate.

### 3.3 A fixed target
The target is the observed station temperature. It is never averaged with the POWER temperature, so the single-source and fused runs predict the same values.

### 3.4 No future values in a feature
A feature gap is filled only forward, for max 3 hours. A mask column records each gap. The target is never filled.

### 3.5 Direct forecasts for all horizons
For an origin t, the targets are t+1 to t+24 in one output. A test checks that target h of each sample is the observed temperature h hours after the origin.

### 3.6 Real validation and baselines first
The split is by year: train, validation and test. Ridge selects alpha, boosting selects its iteration count and the LSTM stops early on the validation years. Each model is scored against persistence.

### 3.7 Train-only preprocessing
Imputation and scaling are steps of an sklearn `Pipeline`, fit on the train samples. The LSTM computes its medians and scaling from the train samples.

---

## 4. The end-to-end workflow

### 4.1 Full flow

```mermaid
flowchart TD
    OW[/"OpenWeather CSV: dt in UTC,<br/>kelvin, hPa"/] --> S["read_openweather_csv:<br/>°C, kPa, sea level and surface apart"]
    PW[/"POWER CSV: LST header, -999"/] --> P["read_power_csv:<br/>UTC shift, -999 to NaN"]
    API[/"POWER API<br/>time-standard=UTC"/] --> CA[("cache/<br/>power_lat_lon_start_end_UTC.csv")]
    CA --> P
    S --> G["build_dataset:<br/>UTC hourly grid, common range"]
    P --> G
    G --> Q["Quality control:<br/>physical limits to NaN"]
    Q --> F["Fusion rules:<br/>fz_rh_pct, fz_rain_mm, fz_sfc_kpa"]
    F --> M["Forward fill 3 h<br/>+ mask columns, target not filled"]
    M --> FT[("data/fused.csv")]
    FT --> W["make_samples: window 24 h,<br/>targets t+1 to t+24"]
    W --> SP["split_by_year + embargo"]
    SP --> B["Baselines: persistence,<br/>seasonal-naive, climatology"]
    SP --> L["ridge, gbm, lstm<br/>station and fused features"]
    B --> E["MAE, RMSE, skill<br/>for each horizon"]
    L --> E
    E --> A[/"Ablation: RMSE fused minus station,<br/>day-block bootstrap interval"/]
    A --> HUMAN{{"HUMAN<br/>analyst reads skill and ablation<br/>before a feature set is used"}}
    FT --> FC[/"forecast: ridge,<br/>next 24 hours"/]

    classDef human fill:#fff3cd,stroke:#b8901f,color:#3d2f00,font-weight:bold
    class HUMAN human
```

### 4.2 The life cycle of one forecast

```mermaid
stateDiagram-v2
    state "Raw rows in two files" as Raw
    state "Hour on the UTC grid" as Grid
    state "Checked hour" as Checked
    state "Filled features and masks" as Filled
    state "Candidate origin t" as Candidate
    state "Sample: window and 24 targets" as Sample
    state "Removed by the embargo" as Embargoed
    state "Train or validation sample" as Fit
    state "Test sample" as Test
    state "Forecast for 24 horizons" as Forecast
    [*] --> Raw
    Raw --> Grid: loaders, LST shifted to UTC
    Grid --> Checked: values outside limits to NaN
    Checked --> Filled: fusion rules, forward fill 3 h
    Filled --> Candidate: make_samples
    Candidate --> Dropped: target at t or a future target missing
    Candidate --> Sample: all targets present
    Sample --> Embargoed: last target in the next part
    Sample --> Fit: origin year up to SKYFUSION_VAL_END
    Sample --> Test: origin year after SKYFUSION_VAL_END
    Fit --> [*]: fit on train, select on validation
    Test --> Forecast: predict change, add last value
    Forecast --> Scored: MAE, RMSE, skill
    Scored --> [*]
    Dropped --> [*]
    Embargoed --> [*]
```

1. Read the station file and change kelvin to °C and hPa to kPa.
2. Read the POWER file, shift LST to UTC and change `-999` to NaN.
3. Put both tables on one hourly UTC grid.
4. Set values outside physical limits to NaN.
5. Average the like-for-like columns and fill feature gaps forward for max 3 hours.
6. Take the last 24 hours before the origin as the input window.
7. Predict the change from the last observed temperature for all 24 horizons.
8. Add the last observed temperature to get the forecast.

### 4.3 Who does which step

```mermaid
sequenceDiagram
    autonumber
    actor A as Analyst
    participant CLI as skyfusion CLI
    participant ING as ingest.py
    participant NASA as NASA POWER API
    participant FUS as fuse.build_dataset
    participant EVA as evaluate.run_experiment
    participant WIN as windows.py
    participant MOD as Models

    A->>CLI: skyfusion fetch-power --start --end
    CLI->>ING: fetch_power(lat, lon, start, end, cache_dir)
    ING->>NASA: GET hourly point, format CSV, time-standard UTC
    NASA-->>ING: CSV text, or JSON error
    ING-->>CLI: UTC table, file in cache/
    A->>CLI: skyfusion build-dataset --station --power --out
    CLI->>ING: read_openweather_csv, read_power_files
    ING-->>CLI: two UTC tables in SI units
    CLI->>FUS: build_dataset(station, power)
    FUS-->>CLI: fused table and QCReport
    CLI-->>A: data/fused.csv
    A->>CLI: skyfusion evaluate --dataset data/fused.csv
    CLI->>EVA: run_experiment(df, window, horizon, years)
    EVA->>WIN: make_samples and split_by_year, station and fused
    loop each feature set and each model
        EVA->>MOD: fit(train, val)
        EVA->>MOD: predict(test)
        MOD-->>EVA: 24 forecasts for each sample
    end
    EVA->>EVA: add_skill, block_bootstrap_diff
    EVA-->>CLI: rows and ablation
    CLI-->>A: MAE and skill table, ablation intervals
```

---

## 5. Ingestion and UTC alignment

**Purpose.** Read each source in its real format and return SI units on a UTC index.

```mermaid
flowchart TD
    TXT[/"POWER CSV text"/] --> HDR["_split_header:<br/>text before -END HEADER-"]
    HDR --> STD{"time_standard given,<br/>or header says LST or UTC?"}
    STD -- "no" --> E1[/"IngestError"/]
    STD -- "yes" --> COLS{"YEAR, MO, DY, HR<br/>columns present?"}
    COLS -- "no" --> E2[/"IngestError: columns missing"/]
    COLS -- "yes" --> SHIFT["Local time minus offset:<br/>round of lon / 15 for LST, 0 for UTC"]
    SHIFT --> FILL["-999 to NaN"]
    FILL --> DUP["Drop duplicate hours,<br/>sort by time"]
    DUP --> OUT[/"POWER table, UTC index"/]
```

`fetch_power()` downloads one POWER request in UTC and keeps it in the cache.

```mermaid
flowchart TD
    REQ[/"lat, lon, start, end<br/>YYYYMMDD"/] --> KEY["Cache file name:<br/>power_lat_lon_start_end_UTC.csv"]
    KEY --> HIT{"File in<br/>SKYFUSION_CACHE_DIR?"}
    HIT -- "yes" --> READ["read_power_csv,<br/>time standard UTC"]
    HIT -- "no" --> GET["GET power_url: 7 parameters,<br/>format CSV, time-standard UTC,<br/>timeout 120 s"]
    GET --> JS{"Answer starts<br/>with a JSON object?"}
    JS -- "yes" --> ERR[/"IngestError with the API messages,<br/>no cache file"/]
    JS -- "no" --> SAVE["Write the cache file"]
    SAVE --> READ
    READ --> OUT[/"POWER table, UTC index"/]
```

`read_openweather_csv()` changes the station export to SI units.

```mermaid
flowchart LR
    CSV[/"OpenWeather History Bulk CSV"/] --> T{"dt or dt_iso?"}
    T -- "dt" --> UX["Unix seconds to UTC"]
    T -- "dt_iso only" --> ISO["Parse the ISO text as UTC"]
    T -- "neither" --> ERR[/"IngestError"/]
    UX --> CONV["temp, dew_point, feels_like:<br/>kelvin minus 273.15"]
    ISO --> CONV
    CONV --> PR["pressure to slp_kpa,<br/>grnd_level to sfc_kpa, hPa / 10"]
    PR --> MAP["humidity, wind_speed,<br/>rain_1h, clouds_all renamed"]
    MAP --> RAIN["Empty rain_mm to 0"]
    RAIN --> OUT[/"Station table, UTC index"/]
```

| Source | Time | Conversion |
|---|---|---|
| POWER file | `in LST` or `in UTC` from the header | LST − round(lon / 15) h → UTC. `-999` → NaN |
| POWER API | `time-standard=UTC` in the request | Cached as `cache/power_<lat>_<lon>_<start>_<end>_UTC.csv` |
| OpenWeather | `dt` in Unix seconds (UTC), else `dt_iso` | Kelvin → °C, hPa → kPa, empty `rain_1h` → 0 |

**Station columns after loading**

| Column | Meaning |
|---|---|
| `temp_c`, `dew_point_c`, `feels_like_c` | Temperatures. `feels_like` is a different quantity and is not mixed with `temp` |
| `slp_kpa` | Sea-level pressure (`pressure`) |
| `sfc_kpa` | Surface pressure (`grnd_level`), only if the export has it |
| `rh_pct`, `wind10_ms`, `rain_mm`, `clouds_pct` | Humidity, wind at 10 m, rain, cloud cover |

**Rules**

- If the POWER header says neither LST nor UTC, the loader stops with an error.
- A POWER API answer in JSON is an error message. The client raises it and writes nothing to the cache.

---

## 6. Quality control and fusion

**Purpose.** Make one clean table with the target, the features and the masks.

```mermaid
flowchart TD
    ST[/"Station table"/] --> UTC{"Both indexes<br/>timezone-aware UTC?"}
    PW[/"POWER table, optional"/] --> UTC
    UTC -- "no" --> ERR[/"ValueError"/]
    UTC -- "yes" --> PFX["st_ prefix, drop st_temp_min_c<br/>and st_temp_max_c. POWER names to pw_"]
    PFX --> GRID["Hourly UTC grid over the<br/>range that both sources cover"]
    GRID --> QC["_qc: value outside LIMITS<br/>to NaN, count each column"]
    QC --> TM["Count the hours with<br/>no st_temp_c"]
    TM --> FZ["FUSION_RULES: mean of the<br/>available station and POWER values"]
    FZ --> GAP{"Feature column<br/>has gaps?"}
    GAP -- "yes" --> MASK["Add the _missing mask,<br/>ffill with limit 3"]
    GAP -- "no" --> TGT
    MASK --> TGT["target_temp_c = st_temp_c,<br/>not filled"]
    TGT --> OUT[/"Fused table and QCReport"/]
```

**Procedure**

1. Prefix the station columns with `st_` and the POWER columns with `pw_`.
2. Make an hourly UTC grid over the time range that both sources cover.
3. Set each value outside its physical limits to NaN and count it.
4. Make the fused columns of the fusion rules (mean of the available values).
5. For each feature column with gaps, add a `*_missing` mask and fill forward for max 3 hours.
6. Copy the unfilled `st_temp_c` to `target_temp_c`.

**Fusion rules**

| Fused column | Station column | POWER column | Why |
|---|---|---|---|
| `fz_rh_pct` | `st_rh_pct` | `pw_rh_pct` | Both are relative humidity near 2 m |
| `fz_rain_mm` | `st_rain_mm` | `pw_rain_mm` | Both are mm in one hour |
| `fz_sfc_kpa` | `st_sfc_kpa` | `pw_sfc_kpa` | Both are surface pressure. Only made if the station has `grnd_level` |

Not fused: `st_slp_kpa` with `pw_sfc_kpa` (sea level against surface), `st_wind10_ms` with `pw_wind2_ms` (10 m against 2 m), and `st_temp_c` with `pw_t2m_c` (the target).

```mermaid
flowchart LR
    subgraph ST["Station columns"]
        SRH["st_rh_pct"]
        SRA["st_rain_mm"]
        SSF["st_sfc_kpa<br/>only with grnd_level"]
        SSL["st_slp_kpa"]
        SWI["st_wind10_ms"]
        STE["st_temp_c"]
    end
    subgraph PWC["POWER columns"]
        PRH["pw_rh_pct"]
        PRA["pw_rain_mm"]
        PSF["pw_sfc_kpa"]
        PWI["pw_wind2_ms"]
        PTE["pw_t2m_c"]
    end
    SRH --> FRH["fz_rh_pct"]
    PRH --> FRH
    SRA --> FRA["fz_rain_mm"]
    PRA --> FRA
    SSF --> FSF["fz_sfc_kpa"]
    PSF --> FSF
    STE --> TGT[("target_temp_c<br/>never averaged")]
    SSL -. "not fused" .- PSF
    SWI -. "not fused" .- PWI
    STE -. "not fused" .- PTE
```

**Physical limits**

| Quantity | Limits |
|---|---|
| Temperature | −60 to 60 °C |
| Relative humidity | 0 to 100 % |
| Sea-level pressure | 85 to 110 kPa |
| Surface pressure | 50 to 110 kPa |
| Wind | 0 to 80 m/s |
| Rain | 0 to 300 mm/h |
| Shortwave and longwave irradiance | 0 to 1500 and 0 to 800 W/m² |

---

## 7. Samples, split and models

**Purpose.** Train and score models on samples that do not leak across time.

```mermaid
flowchart TD
    DF[/"Fused table and a feature set:<br/>station or fused"/] --> REG{"target_temp_c present and<br/>a regular hourly grid?"}
    REG -- "no" --> ERR[/"KeyError or ValueError"/]
    REG -- "yes" --> POS["Origins t from hour W-1<br/>to the last hour minus H"]
    POS --> WIN["sliding_window_view:<br/>X_seq hours t-23 to t,<br/>y_hist, Y hours t+1 to t+24"]
    WIN --> OK{"Target at t and<br/>all H targets present?"}
    OK -- "no" --> DROP["Drop the origin"]
    OK -- "yes" --> CAL["Calendar: hour and day-of-year,<br/>sine and cosine"]
    CAL --> S[/"Samples: origins, X_seq,<br/>calendar, y_hist, Y"/]
```

The diagram shows how `split_by_year()` puts each sample in one part.

```mermaid
flowchart TD
    S[/"Sample with origin t"/] --> Y{"Year of t"}
    Y -- "up to SKYFUSION_TRAIN_END" --> E1{"Last target hour t+H<br/>also in a train year?"}
    E1 -- "yes" --> TR[/"train"/]
    E1 -- "no" --> EMB[/"Removed: embargo"/]
    Y -- "up to SKYFUSION_VAL_END" --> E2{"Last target hour t+H<br/>also in a validation year?"}
    E2 -- "yes" --> VA[/"validation"/]
    E2 -- "no" --> EMB
    Y -- "later" --> TE[/"test"/]
    TR --> CHK{"A part is empty?"}
    VA --> CHK
    TE --> CHK
    CHK -- "yes" --> ERR[/"ValueError: empty split"/]
```

**Procedure**

1. For each origin t, take the window t−23 to t of the feature columns and the targets t+1 to t+24.
2. Keep a sample only if the target at t and all 24 targets exist.
3. Add hour-of-day and day-of-year sine and cosine values of the origin.
4. Put a sample in train, validation or test by the year of its origin.
5. Remove train and validation samples whose last target hour is in the next part (embargo).
6. Fit each model on train, select on validation, and score on test.

**Models**

| Model | Prediction | Selection on validation |
|---|---|---|
| `persistence` | Last observed temperature for all horizons | none |
| `seasonal-naive` | Temperature at the same hour on the day before | none |
| `climatology` | Train mean for each month and UTC hour | none |
| `ridge` | Change from the last value: imputer, scaler, ridge on the flat 24 h window | alpha ∈ {0.1, 1, 10, 100} |
| `gbm` | One histogram gradient boosting model for each horizon, last 6 hours of inputs | iteration count (max 150) |
| `lstm` | LSTM (64 units) with a linear head for all horizons, mask flags as extra inputs | early stopping, patience 4 |

The ablation trains `ridge`, `gbm` and `lstm` two times: with the `station` feature set and with the `fused` feature set. The samples, origins and targets are identical.

The learned models predict the change from the last observed temperature. Each one uses the validation samples in a different way.

```mermaid
flowchart TD
    TR[/"Train samples"/] --> R["Change targets:<br/>Y minus last observed value"]
    R --> M{"Model"}
    M -- "ridge" --> RP["For each alpha in 0.1, 1, 10, 100:<br/>median imputer, scaler, ridge<br/>on the flat 24 h window"]
    RP --> RS["Keep the alpha with the<br/>lowest validation RMSE"]
    M -- "gbm" --> GP["For each horizon: boosting on<br/>the last 6 hours, 150 iterations"]
    GP --> GS["staged_predict on validation:<br/>keep the best iteration count"]
    M -- "lstm" --> LP["Train medians and scaling,<br/>missing flags, calendar.<br/>LSTM 64 units, linear head"]
    LP --> LS["Early stopping on validation loss,<br/>patience 4, keep the best weights"]
    RS --> P["predict: change + last<br/>observed value"]
    GS --> P
    LS --> P
    P --> OUT[/"24 forecasts for each test sample"/]
```

`run_experiment()` runs the comparison on the same test samples.

```mermaid
flowchart LR
    DF[/"Fused table"/] --> FS["feature_sets:<br/>station, fused"]
    FS --> SM["make_samples and<br/>split_by_year for each set"]
    SM --> SAME{"Same test origins<br/>in both sets?"}
    SAME -- "no" --> ERR[/"AssertionError"/]
    SAME -- "yes" --> ST["station: 3 baselines,<br/>ridge, gbm, lstm with --lstm"]
    SAME -- "yes" --> FU["fused: ridge, gbm,<br/>lstm with --lstm"]
    ST --> SC["score: MAE, RMSE<br/>for each horizon"]
    FU --> SC
    SC --> SK["add_skill"]
    SK --> AB[/"Ablation at h = 1, 6, 24,<br/>or h = 1 and H if H below 24"/]
```

---

## 8. The decision rules

| Setting | Value | Where |
|---|---|---|
| Window | 24 hours (`SKYFUSION_WINDOW`) | `make_samples` |
| Horizon | 24 hours (`SKYFUSION_HORIZON`, max 72) | `make_samples` |
| Train years | up to `SKYFUSION_TRAIN_END` (default 2019) | `split_by_year` |
| Validation years | up to `SKYFUSION_VAL_END` (default 2021) | `split_by_year` |
| Test years | later than the validation years | `split_by_year` |
| Forward fill limit | 3 hours, features only | `build_dataset` |
| LST offset | round(lon / 15) hours | `Settings.lst_offset_hours`, `parse_power_text` |
| Skill | 1 − RMSE / RMSE of persistence, at each horizon | `evaluate.add_skill` |
| Ablation interval | 500 resamples of whole days, 95 % percentile | `block_bootstrap_diff` |

The diagram shows how skill and the ablation interval are calculated.

```mermaid
flowchart LR
    P[/"Test predictions of<br/>each model and feature set"/] --> RM["RMSE at each horizon"]
    PE[/"persistence RMSE,<br/>station features"/] --> SK["skill = 1 − RMSE /<br/>RMSE of persistence"]
    RM --> SK
    PF[/"fused and station predictions<br/>of one learned model"/] --> DAY["Squared errors at horizon h,<br/>summed for each UTC day"]
    DAY --> BS["500 resamples of whole days,<br/>seed SKYFUSION_SEED"]
    BS --> DIFF["RMSE fused minus<br/>RMSE station for each resample"]
    DIFF --> CI[/"Point value and<br/>2.5 and 97.5 percentiles"/]
```

---

## 9. Data and file map

| Path | Committed? | Contents |
|---|---|---|
| `data/README.md` | Yes | Sources, terms, columns |
| `data/station_openweather.csv` | No (git ignores it) | Licensed station export |
| `data/synthetic/` | No (git ignores it) | Output of `skyfusion synth` |
| `data/fused.csv` | No (git ignores it) | Output of `skyfusion build-dataset` |
| `cache/` | No (git ignores it) | POWER downloads |
| `.env` | No (git ignores it) | Local settings |

---

## 10. How to run skyfusion

### 10.1 Prerequisites

| Need | For |
|---|---|
| Python 3.11+ | All components |
| Network access | `fetch-power` only |
| An OpenWeather History Bulk export | Real station data |
| Extra `lstm` | The LSTM (`evaluate --lstm`) |

### 10.2 Installation

```bash
git clone https://github.com/KrishnaAnnavaram/skyfusion.git
cd skyfusion
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 10.3 Run skyfusion

```bash
# Offline demo: synthetic station (UTC) and POWER (LST) files for 2018-2022
skyfusion synth --out data/synthetic
skyfusion qc --station data/synthetic/station_openweather.csv --power data/synthetic/power_lst.csv
skyfusion build-dataset --station data/synthetic/station_openweather.csv --power data/synthetic/power_lst.csv --out data/fused.csv
skyfusion evaluate --dataset data/fused.csv
skyfusion forecast --dataset data/fused.csv --fused

# Optional LSTM
pip install -e ".[lstm]"
skyfusion evaluate --dataset data/fused.csv --lstm

# Real data
skyfusion fetch-power --start 20010101 --end 20241231
skyfusion build-dataset --station data/station_openweather.csv --power cache/*.csv --out data/fused.csv
```

The diagram shows the order of the commands and the files that connect them.

```mermaid
flowchart LR
    INS["pip install -e .[dev]"] --> SYN["skyfusion synth"]
    SYN --> RAW[("data/synthetic/<br/>station_openweather.csv,<br/>power_lst.csv")]
    FP["skyfusion fetch-power"] --> CA[("cache/power_*_UTC.csv")]
    RAW --> QC["skyfusion qc<br/>prints the QCReport"]
    RAW --> BD["skyfusion build-dataset"]
    CA --> BD
    BD --> FT[("data/fused.csv")]
    FT --> EV["skyfusion evaluate<br/>--lstm, --fast, --json"]
    FT --> FC["skyfusion forecast<br/>--fused"]
```

The `forecast` command makes one forecast from the last window of the table.

```mermaid
flowchart TD
    FT[/"data/fused.csv"/] --> FS{"--fused?"}
    FS -- "yes" --> FU["fused feature set"]
    FS -- "no" --> ST["station feature set"]
    FU --> SM["make_samples, split_by_year"]
    ST --> SM
    SM --> FIT["RidgeDirect: fit on train,<br/>alpha from validation"]
    SM --> LAST["Sample with the<br/>latest origin"]
    FIT --> PR["predict: change +<br/>last observed value"]
    LAST --> PR
    PR --> OUT[/"Temperature for each of<br/>the next H hours, in UTC"/]
```

### 10.4 Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `SKYFUSION_LAT` | POWER client | Latitude, default 32.8998 |
| `SKYFUSION_LON` | POWER client, LST shift | Longitude, default −97.0403 |
| `SKYFUSION_CACHE_DIR` | POWER client | Cache folder, default `cache` |
| `SKYFUSION_TRAIN_END` | split | Last train year, default 2019 |
| `SKYFUSION_VAL_END` | split | Last validation year, default 2021 |
| `SKYFUSION_HORIZON` | samples | Hours ahead, default 24 |
| `SKYFUSION_WINDOW` | samples | Input hours, default 24 |
| `SKYFUSION_SEED` | models | Random seed, default 7 |

skyfusion needs no API key. Local settings are only in a `.env` file. Git ignores this file.

---

## 11. How to extend skyfusion

| You want to… | Do this | Code change? |
|---|---|---|
| Use another site | Set `SKYFUSION_LAT` and `SKYFUSION_LON`, run `fetch-power` | No |
| Change the split years | Set `SKYFUSION_TRAIN_END` and `SKYFUSION_VAL_END` | No |
| Add a fusion rule | Add a pair of the same quantity to `FUSION_RULES` | Small |
| Add a quality limit | Add an entry to `LIMITS` | Small |
| Add a model | Subclass `Model` with `fit(train, val)` and `predict(samples)` and add it to `default_models` | Small |

---

## 12. Validation results

All numbers come from synthetic data (2018 to 2022, one hidden truth). They are synthetic results, not results for a real site.

| Validation | Result | Command |
|---|---|---|
| Unit tests | CI installs only `.[dev]`: **25 passed**, 1 skipped (`lstm` extra, PyTorch). With the extras: 26 passed | `pytest -q` |
| UTC alignment | Station and POWER temperatures correlate best at lag 0 | `pytest tests/test_ingest_fuse.py` |
| Data | 43,824 hours, 549 hours without a station temperature (1 % random gaps and a 5-day outage) | `skyfusion build-dataset` |

**Test year 2022: 6,825 samples (train 2018-2019: 14,113, validation 2020-2021: 13,226). MAE in °C and skill**

| Model | Features | MAE h=1 | Skill h=1 | MAE h=6 | Skill h=6 | MAE h=24 | Skill h=24 |
|---|---|---|---|---|---|---|---|
| `persistence` | station | 0.839 | 0.000 | 3.984 | 0.000 | 1.318 | 0.000 |
| `seasonal-naive` | station | 1.323 | −0.656 | 1.366 | 0.622 | 1.318 | 0.000 |
| `climatology` | station | 2.584 | −2.173 | 2.570 | 0.312 | 2.572 | −0.913 |
| `ridge` | station | 0.456 | 0.432 | 0.996 | 0.732 | 1.239 | 0.065 |
| `gbm` | station | 0.462 | 0.428 | 1.001 | 0.729 | 1.276 | 0.031 |
| `lstm` | station | 0.458 | 0.430 | 0.923 | 0.750 | 1.251 | 0.055 |
| `ridge` | fused | **0.422** | **0.477** | 0.964 | 0.741 | **1.235** | **0.067** |
| `gbm` | fused | 0.435 | 0.460 | **0.869** | **0.753** | 1.263 | 0.043 |
| `lstm` | fused | 0.514 | 0.344 | 1.313 | 0.550 | 1.276 | 0.038 |

**Ablation: RMSE(fused) − RMSE(station), 95 % day-block bootstrap interval (negative means fusion helps)**

| Model | h=1 | h=6 | h=24 |
|---|---|---|---|
| `ridge` | −0.045 (−0.051 to −0.040) | −0.044 (−0.061 to −0.027) | −0.003 (−0.014 to +0.008) |
| `gbm` | −0.033 (−0.041 to −0.025) | −0.109 (−0.183 to −0.035) | −0.019 (−0.037 to +0.000) |
| `lstm` | +0.087 (+0.062 to +0.116) | +0.928 (+0.696 to +1.176) | +0.028 (+0.004 to +0.050) |

What the numbers show:

- The learned models beat persistence at every horizon. At 24 h, persistence equals the same hour yesterday, and the gain is small.
- POWER features improve ridge and boosting at 1 h and 6 h. The intervals do not include 0.
- The fused LSTM is worse. A possible cause: the last 60 days of the test year have POWER irradiance fill values, and the train years have none. Problem 3 in section 13 gives the action.
- The synthetic generator lets clouds, which only POWER sees, change the temperature some hours later. So the size of the fusion gain is a property of the generator.
- The prototype reported that fusion reduced the MSE from 0.91 to 0.24. That is a prototype result, not reproduced here, and it compared different targets.

---

## 13. Known problems

Read these problems before you use skyfusion in production.

| # | Area | Problem | Impact and action |
|---|---|---|---|
| 1 | Data | All results come from synthetic data | Run `fetch-power` and a real station export, then `evaluate` |
| 2 | Station | The real OpenWeather export is a licensed file and is not in the repository | Buy or export it for your site. See `data/README.md` |
| 3 | LSTM | Gaps that occur only in the test years confuse the LSTM | Add training samples with gaps (mask augmentation), or drop columns with long gaps |
| 4 | Timing | POWER reanalysis is published with a delay | For real-time use, use only features that exist at the forecast time |
| 5 | Target | Only temperature is forecast | Add humidity as a second target with the same code |
| 6 | Speed | Gradient boosting trains one model for each horizon | Use fewer horizons or `--fast` for quick checks |
| 7 | Sea-level pressure | The station gives only sea-level pressure in many exports | Surface pressure is then POWER only. No fused pressure is made |

---

## 14. Key points

1. **One UTC grid.** POWER LST is shifted, and a test checks the alignment at lag 0.
2. **Only like-for-like fusion.** Sea-level pressure, 10 m wind and the target are never averaged with POWER.
3. **No future values.** Features fill only forward, the target never, and an embargo separates the splits.
4. **Direct 24-hour forecasts.** Each horizon has its own scored output, with no recursive loop.
5. **Baselines and a fair ablation.** Skill is relative to persistence, and the two feature sets use identical samples.
6. **Everything runs offline.** The synthetic files use the real raw formats, and the 25 CI tests need no network.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Station** | The OpenWeather observations of the site (`st_*` columns) |
| **POWER** | NASA POWER hourly data (`pw_*` columns) |
| **UTC grid** | The hourly UTC index that all columns share |
| **LST** | Local standard time of a POWER file, UTC + round(lon / 15) hours |
| **Fill value** | The POWER code `-999` for a missing value |
| **Fusion rule** | A pair of same-quantity columns that skyfusion averages |
| **Target** | The observed station temperature `target_temp_c` |
| **Mask column** | A `*_missing` column that is 1 where a value was missing |
| **Origin** | The last observed hour of a sample |
| **Window** | The input hours that end at the origin |
| **Horizon** | The number of hours after the origin |
| **Direct forecast** | One model output for all horizons |
| **Embargo** | The removal of samples whose targets cross into the next split part |
| **Baseline** | Persistence, seasonal naive or climatology |
| **Skill** | 1 − RMSE / RMSE of persistence |
| **Ablation** | The comparison of the station and fused feature sets on identical samples |
| **MAE** | Mean absolute error in °C |
| **RMSE** | Root mean squared error in °C |

---

## 16. License

[MIT](LICENSE) © 2026 Krishna Annavaram
