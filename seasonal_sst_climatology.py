# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# SEASONAL CLIMATOLOGY + P90 COMBO PLOT
#
# Region:
#   0-30°N, 60-10°W
#
# Baseline:
#   1982-2010
#
# Method:
#   1. Calculate area-weighted regional daily SST
#   2. Calculate daily climatological mean
#   3. Calculate daily P90 using ±5-day window
#   4. Apply 31-day circular smoothing
#   5. Average the DAILY climatology and DAILY P90
#      into DJF, MAM, JJA and SON
#
# Plot:
#   Bars = Seasonal climatological mean SST
#   Line = Seasonal P90
#
# No CSV
# No figure saved
# ==============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ==============================================================
# 1. SETTINGS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0

BASE_START = "1982-01-01"
BASE_END   = "2010-12-31"

PERCENTILE = 90

HALF_WINDOW = 5

SMOOTH_WINDOW = 31


# ==============================================================
# 2. PREPROCESS OISST
# ==============================================================

def preprocess(ds):

    rename = {}

    for old, new in {
        "latitude": "lat",
        "longitude": "lon",
        "Time": "time",
        "TIME": "time"
    }.items():

        if old in ds.coords or old in ds.dims:
            rename[old] = new

    if rename:
        ds = ds.rename(rename)

    # Remove singleton vertical dimensions
    for dim in [
        "zlev",
        "depth",
        "lev",
        "level"
    ]:

        if (
            dim in ds.dims
            and ds.sizes[dim] == 1
        ):
            ds = ds.squeeze(
                dim,
                drop=True
            )

    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found."
        )

    ds = ds[["sst"]]

    # Convert longitude to -180...180
    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    # Tropical North East Atlantic
    ds = ds.sel(
        lat=slice(
            LAT_MIN,
            LAT_MAX
        ),
        lon=slice(
            LON_MIN,
            LON_MAX
        )
    )

    return ds


# ==============================================================
# 3. CLIMATOLOGICAL DAY
# ==============================================================

def get_clim_day(dates):

    """
    Map every date onto leap year 2000.

    Jan 1  = 1
    Feb 29 = 60
    Dec 31 = 366
    """

    dates = pd.DatetimeIndex(
        dates
    )

    reference_dates = pd.to_datetime(
        {
            "year": np.full(
                len(dates),
                2000
            ),

            "month": dates.month,

            "day": dates.day
        }
    )

    return (
        pd.DatetimeIndex(
            reference_dates
        )
        .dayofyear
        .to_numpy()
    )


# ==============================================================
# 4. CIRCULAR SMOOTHING
# ==============================================================

def circular_smooth(
    values,
    window=31
):

    half = (
        window // 2
    )

    extended = np.concatenate(
        [
            values[-half:],
            values,
            values[:half]
        ]
    )

    smoothed = (
        pd.Series(
            extended
        )
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )

    return smoothed[
        half:
        half + len(values)
    ]


# ==============================================================
# 5. FIND OISST FILES
# ==============================================================

files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "*_oisst.nc"
        )
    )
)

if not files:

    files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "*.nc"
            )
        )
    )

if not files:

    raise FileNotFoundError(
        f"No NetCDF files found in:\n"
        f"{DATA_DIR}"
    )


print("=" * 80)

print(
    "TROPICAL NORTH EAST ATLANTIC "
    "SEASONAL CLIMATOLOGY + P90"
)

print("=" * 80)

print(
    f"\nSST files found: "
    f"{len(files):,}"
)


# ==============================================================
# 6. OPEN DATA
# ==============================================================

print(
    "\nOpening OISST data..."
)

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    preprocess=preprocess,
    parallel=False,
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="outer",
    engine="netcdf4",
    chunks={
        "time": 365
    }
)

ds = ds.sortby(
    "time"
)

sst = ds[
    "sst"
]


# ==============================================================
# 7. NORMALIZE TIME
# ==============================================================

time_index = pd.DatetimeIndex(
    sst.time.values
).normalize()

sst = sst.assign_coords(
    time=time_index
)

keep = np.where(
    ~time_index.duplicated(
        keep="first"
    )
)[0]

sst = sst.isel(
    time=keep
)


# ==============================================================
# 8. SELECT 1982-2010
# ==============================================================

sst = sst.sel(
    time=slice(
        BASE_START,
        BASE_END
    )
)

dates = pd.DatetimeIndex(
    sst.time.values
)

print(
    f"\nClimatology period: "
    f"{dates[0].date()} to "
    f"{dates[-1].date()}"
)

print(
    f"Number of daily observations: "
    f"{len(dates):,}"
)

print(
    f"Grid: "
    f"{sst.sizes['lat']} × "
    f"{sst.sizes['lon']}"
)


# ==============================================================
# 9. UNIT CHECK
# ==============================================================

sample = float(
    sst.isel(
        time=slice(
            0,
            10
        )
    )
    .mean(
        skipna=True
    )
    .compute()
)

if sample > 100:

    print(
        "\nConverting SST from Kelvin to °C..."
    )

    sst = (
        sst
        -
        273.15
    )

else:

    print(
        "\nSST already appears to be °C."
    )


# ==============================================================
# 10. AREA-WEIGHTED REGIONAL DAILY SST
# ==============================================================

print(
    "\nCalculating area-weighted regional daily SST..."
)


# Latitude area weights
weights = np.cos(
    np.deg2rad(
        sst.lat
    )
)


# First average longitude
# Then latitude with cosine weighting

regional_daily = (
    sst
    .mean(
        dim="lon",
        skipna=True
    )
    .weighted(
        weights
    )
    .mean(
        dim="lat",
        skipna=True
    )
)


print(
    "Loading regional daily SST series..."
)

regional_daily = (
    regional_daily
    .compute()
)


regional_dates = pd.DatetimeIndex(
    regional_daily.time.values
)

regional_values = np.asarray(
    regional_daily.values,
    dtype=float
).reshape(-1)


# ==============================================================
# 11. CREATE COMPLETE DAILY SERIES
# ==============================================================

series = pd.Series(
    regional_values,
    index=regional_dates
)

series = series[
    ~series.index.duplicated(
        keep="first"
    )
]

full_dates = pd.date_range(
    BASE_START,
    BASE_END,
    freq="D"
)

series = series.reindex(
    full_dates
)


baseline_dates = (
    series.index
)

baseline_values = (
    series.values
)


print(
    f"\nValid regional SST days: "
    f"{np.sum(np.isfinite(baseline_values)):,}"
)


# ==============================================================
# 12. CLIMATOLOGICAL DAY
# ==============================================================

clim_day = get_clim_day(
    baseline_dates
)


# ==============================================================
# 13. CALCULATE DAILY CLIMATOLOGY AND P90
# ==============================================================

print(
    "\nCalculating daily climatological mean and P90..."
)


daily_climatology = np.full(
    366,
    np.nan,
    dtype=float
)

daily_p90 = np.full(
    366,
    np.nan,
    dtype=float
)


for day in range(
    1,
    367
):

    # Circular distance
    distance = np.abs(
        clim_day
        -
        day
    )

    distance = np.minimum(
        distance,
        366
        -
        distance
    )


    # ±5-day window
    selected = (
        distance
        <=
        HALF_WINDOW
    )


    day_values = baseline_values[
        selected
    ]


    day_values = day_values[
        np.isfinite(
            day_values
        )
    ]


    if len(
        day_values
    ) > 0:

        # Daily climatological mean
        daily_climatology[
            day - 1
        ] = np.mean(
            day_values
        )


        # Daily P90
        daily_p90[
            day - 1
        ] = np.percentile(
            day_values,
            PERCENTILE
        )


    if (
        day == 1
        or
        day % 30 == 0
        or
        day == 366
    ):

        print(
            f"Day "
            f"{day:03d}/366 completed"
        )


# ==============================================================
# 14. 31-DAY CIRCULAR SMOOTHING
# ==============================================================

print(
    "\nApplying 31-day circular smoothing..."
)


daily_climatology = circular_smooth(
    daily_climatology,
    SMOOTH_WINDOW
)

daily_p90 = circular_smooth(
    daily_p90,
    SMOOTH_WINDOW
)


# ==============================================================
# 15. VALIDATE DAILY CURVES
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "DAILY CLIMATOLOGY / P90"
)

print(
    "=" * 80
)


print(
    f"Climatology minimum: "
    f"{np.nanmin(daily_climatology):.2f} °C"
)

print(
    f"Climatology mean: "
    f"{np.nanmean(daily_climatology):.2f} °C"
)

print(
    f"Climatology maximum: "
    f"{np.nanmax(daily_climatology):.2f} °C"
)


print(
    f"\nP90 minimum: "
    f"{np.nanmin(daily_p90):.2f} °C"
)

print(
    f"P90 mean: "
    f"{np.nanmean(daily_p90):.2f} °C"
)

print(
    f"P90 maximum: "
    f"{np.nanmax(daily_p90):.2f} °C"
)


# ==============================================================
# 16. ASSIGN EACH CLIMATOLOGICAL DAY TO A SEASON
# ==============================================================

reference_dates = pd.date_range(
    "2000-01-01",
    "2000-12-31",
    freq="D"
)


season_df = pd.DataFrame(
    {
        "date":
            reference_dates,

        "climatology":
            daily_climatology,

        "p90":
            daily_p90
    }
)


month = season_df[
    "date"
].dt.month


season_df["season"] = np.select(

    [
        month.isin(
            [12, 1, 2]
        ),

        month.isin(
            [3, 4, 5]
        ),

        month.isin(
            [6, 7, 8]
        ),

        month.isin(
            [9, 10, 11]
        )
    ],

    [
        "DJF",
        "MAM",
        "JJA",
        "SON"
    ]
)


# ==============================================================
# 17. CALCULATE SEASONAL VALUES FROM DAILY CURVES
# ==============================================================

season_order = [
    "DJF",
    "MAM",
    "JJA",
    "SON"
]


seasonal_climatology = (
    season_df
    .groupby(
        "season"
    )
    ["climatology"]
    .mean()
    .reindex(
        season_order
    )
)


seasonal_p90 = (
    season_df
    .groupby(
        "season"
    )
    ["p90"]
    .mean()
    .reindex(
        season_order
    )
)


# ==============================================================
# 18. PRINT SEASONAL RESULTS
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "SEASONAL CLIMATOLOGY AND P90"
)

print(
    "=" * 80
)


for season in season_order:

    print(
        f"{season}: "
        f"Climatology = "
        f"{seasonal_climatology.loc[season]:.2f} °C | "
        f"P90 = "
        f"{seasonal_p90.loc[season]:.2f} °C"
    )


# ==============================================================
# 19. COMBO PLOT
# ==============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        7
    )
)


x = np.arange(
    len(
        season_order
    )
)


# ==============================================================
# BARS: SEASONAL CLIMATOLOGICAL MEAN
# ==============================================================

bars = ax.bar(
    x,
    seasonal_climatology.values,
    width=0.60,
    label="Seasonal Climatological Mean SST"
)


# ==============================================================
# LINE: SEASONAL P90
# ==============================================================

ax.plot(
    x,
    seasonal_p90.values,
    marker="o",
    linewidth=2.2,
    markersize=7,
    label="Seasonal P90"
)


# ==============================================================
# ADD BAR VALUES
# ==============================================================

for bar, value in zip(
    bars,
    seasonal_climatology.values
):

    ax.text(
        bar.get_x()
        +
        bar.get_width() / 2,

        value + 0.03,

        f"{value:.2f}°C",

        ha="center",

        va="bottom",

        fontsize=10,

        fontweight="bold"
    )


# ==============================================================
# ADD P90 VALUES
# ==============================================================

for position, value in zip(
    x,
    seasonal_p90.values
):

    ax.text(
        position,
        value + 0.05,
        f"{value:.2f}°C",
        ha="center",
        va="bottom",
        fontsize=10
    )


# ==============================================================
# AXES
# ==============================================================

ax.set_xticks(
    x
)

ax.set_xticklabels(
    season_order
)


ax.set_xlabel(
    "Season",
    fontsize=12,
    fontweight="bold"
)


ax.set_ylabel(
    "SST (°C)",
    fontsize=12,
    fontweight="bold"
)


# ==============================================================
# TITLE
# ==============================================================

ax.set_title(
    "Tropical North East Atlantic Seasonal SST Climatology "
    "and P90 (1982–2010)",
    fontsize=14,
    fontweight="bold",
    pad=15
)


# ==============================================================
# Y AXIS RANGE
#
# Do not start at zero because SST values are around 20-30°C.
# ==============================================================

all_values = np.concatenate(
    [
        seasonal_climatology.values,
        seasonal_p90.values
    ]
)


y_min = (
    np.floor(
        np.nanmin(
            all_values
        )
        *
        2
    )
    /
    2
    -
    0.5
)


y_max = (
    np.ceil(
        np.nanmax(
            all_values
        )
        *
        2
    )
    /
    2
    +
    0.5
)


ax.set_ylim(
    y_min,
    y_max
)


# ==============================================================
# GRID + LEGEND
# ==============================================================

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.30
)

ax.legend(
    loc="best"
)


plt.tight_layout()

plt.show()


# ==============================================================
# 20. CLOSE DATASET
# ==============================================================

ds.close()


print(
    "\nSeasonal climatology combo plot completed successfully."
)