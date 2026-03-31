from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("./data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_START = 2007
YEAR_END = 2023
N_CLUSTERS = 5
MIN_NON_NULL_YEARS = 8

CORE_FILES: Dict[str, str] = {
    "co2_total_mmtco2": "co2-emissions/co2-emissions-emissions-mmtco2.csv",
    "gdp_billion_ppp_usd": "energy-intensity/gross-domestic-product-gdp-billion-dollars-at-purchasing-power-parities.csv",
    "population_thousands": "energy-intensity/population-population-people-in-thousands.csv",
    "electricity_generation_bkwh": "electricity/electricity-generation-bkwh.csv",
    "renewables_generation_bkwh": "electricity/renewables-generation-bkwh.csv",
    "fossil_generation_bkwh": "electricity/fossil-fuels-generation-bkwh.csv",
    "hydro_generation_bkwh": "electricity/hydroelectricity-generation-bkwh.csv",
    "wind_generation_bkwh": "electricity/wind-generation-bkwh.csv",
    "solar_generation_bkwh": "electricity/solar-generation-bkwh.csv",
}

EXCLUDED_CODES = {
    "WORL", "OECD", "EU27", "AFRC", "ASIA", "MIDE", "SAMR", "NAMR", "EURA",
    "SCEN", "CEUR", "EEUR", "BALT", "ANZ", "APEC", "BRICS", "ASEAN"
}

EXCLUDED_NAME_PATTERNS = [
    r"\bworld\b",
    r"\boecd\b",
    r"\beu\b",
    r"\bafrica\b",
    r"\basia\b",
    r"\beurope\b",
    r"\bmiddle east\b",
    r"\bnorth america\b",
    r"\bsouth america\b",
    r"\bcentral america\b",
    r"\bformer\b",
    r"\bunion\b",
    r"\bother\b",
    r"\btotal\b",
]

HISTORICAL_ENTITIES = {
    "USSR", "YUGO", "CSK", "EGER", "SCG", "YUG", "SUN", "DDR"
}


# ============================================================
# HELPERS
# ============================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        na_values=["--", "NA", "", " ", "  ", "...", "…"],
        keep_default_na=True
    )


def detect_year_columns(columns: List[str]) -> List[str]:
    year_cols = []
    for c in columns:
        s = str(c).strip()
        if re.fullmatch(r"\d{4}", s):
            year_cols.append(s)
    return year_cols


def standardise_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-specific parser.
    Expected format:
    first column  = Country
    second column = Code
    remaining     = years
    """
    out = df.copy()
    cols = list(out.columns)

    if len(cols) < 3:
        raise ValueError(f"Expected at least 3 columns, got: {cols}")

    out = out.rename(columns={
        cols[0]: "country_name",
        cols[1]: "country_code",
    })

    out["country_name"] = out["country_name"].astype(str).str.strip()
    out["country_code"] = out["country_code"].astype(str).str.strip().str.upper()

    return out


def exclude_non_countries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["country_code"] = out["country_code"].astype(str).str.strip().str.upper()
    out["country_name"] = out["country_name"].astype(str).str.strip()

    out = out[~out["country_code"].isin(EXCLUDED_CODES)]
    out = out[~out["country_code"].isin(HISTORICAL_ENTITIES)]

    lower_names = out["country_name"].str.lower()
    mask = pd.Series(False, index=out.index)

    for pattern in EXCLUDED_NAME_PATTERNS:
        mask = mask | lower_names.str.contains(pattern, regex=True, na=False)

    out = out[~mask]
    out = out[out["country_code"].str.fullmatch(r"[A-Z]{3}", na=False)]

    return out.reset_index(drop=True)


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def pct_change_from_base(current: pd.Series, base: pd.Series) -> pd.Series:
    base = base.replace(0, np.nan)
    return (current - base) / base * 100.0


def verify_core_files(base_dir: Path, core_files: Dict[str, str]) -> None:
    missing = []
    for metric_name, rel_path in core_files.items():
        path = base_dir / rel_path
        if not path.exists():
            missing.append((metric_name, str(path)))
    if missing:
        msg = "Missing configured files:\n"
        for metric_name, path in missing:
            msg += f"  - {metric_name}: {path}\n"
        raise FileNotFoundError(msg)


# ============================================================
# METRIC LOADING
# ============================================================

def load_metric(path: Path, metric_name: str) -> pd.DataFrame:
    df = read_csv_robust(path)
    df = standardise_id_columns(df)

    year_cols = detect_year_columns(df.columns.tolist())
    if not year_cols:
        raise ValueError(f"No year columns detected in {path}")

    long_df = df.melt(
        id_vars=["country_name", "country_code"],
        value_vars=year_cols,
        var_name="year",
        value_name=metric_name
    )

    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df[metric_name] = pd.to_numeric(long_df[metric_name], errors="coerce")

    long_df = long_df[long_df["year"].between(YEAR_START, YEAR_END, inclusive="both")].copy()
    long_df["year"] = long_df["year"].astype(int)

    long_df = exclude_non_countries(long_df)

    long_df = long_df.sort_values(["country_code", "country_name", "year"]).drop_duplicates(
        subset=["country_code", "year"], keep="first"
    )

    long_df[metric_name] = (
        long_df.groupby("country_code", dropna=False)[metric_name]
        .transform(lambda s: s.interpolate(method="linear", limit=2, limit_direction="both"))
    )

    return long_df.reset_index(drop=True)


def build_metric_registry(base_dir: Path, core_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    verify_core_files(base_dir, core_files)

    registry = {}
    for metric_name, rel_path in core_files.items():
        path = base_dir / rel_path
        print(f"\nLoading metric: {metric_name}")
        metric_df = load_metric(path, metric_name)
        print(metric_df[["country_name", "country_code"]].drop_duplicates().head(5))
        print(f"{metric_name}: rows={len(metric_df)}, countries={metric_df['country_code'].nunique()}")
        registry[metric_name] = metric_df

    return registry


# ============================================================
# FEATURE BUILDERS
# ============================================================

def metric_only(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    return df[["country_code", "country_name", "year", metric_name]].copy()


def build_base_panel(registry: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    co2 = metric_only(registry["co2_total_mmtco2"], "co2_total_mmtco2")
    gdp = metric_only(registry["gdp_billion_ppp_usd"], "gdp_billion_ppp_usd")
    pop = metric_only(registry["population_thousands"], "population_thousands")
    elec = metric_only(registry["electricity_generation_bkwh"], "electricity_generation_bkwh")
    ren = metric_only(registry["renewables_generation_bkwh"], "renewables_generation_bkwh")
    fos = metric_only(registry["fossil_generation_bkwh"], "fossil_generation_bkwh")
    hyd = metric_only(registry["hydro_generation_bkwh"], "hydro_generation_bkwh")
    wnd = metric_only(registry["wind_generation_bkwh"], "wind_generation_bkwh")
    sol = metric_only(registry["solar_generation_bkwh"], "solar_generation_bkwh")

    panel = co2.merge(
        gdp[["country_code", "year", "gdp_billion_ppp_usd"]],
        on=["country_code", "year"],
        how="left"
    ).merge(
        pop[["country_code", "year", "population_thousands"]],
        on=["country_code", "year"],
        how="left"
    ).merge(
        elec[["country_code", "year", "electricity_generation_bkwh"]],
        on=["country_code", "year"],
        how="left"
    ).merge(
        ren[["country_code", "year", "renewables_generation_bkwh"]],
        on=["country_code", "year"],
        how="left"
    ).merge(
        fos[["country_code", "year", "fossil_generation_bkwh"]],
        on=["country_code", "year"],
        how="left"
    ).merge(
        hyd[["country_code", "year", "hydro_generation_bkwh"]],
        on=["country_code", "year"],
        how="left"
    ).merge(
        wnd[["country_code", "year", "wind_generation_bkwh"]],
        on=["country_code", "year"],
        how="left"
    ).merge(
        sol[["country_code", "year", "solar_generation_bkwh"]],
        on=["country_code", "year"],
        how="left"
    )

    panel = panel.sort_values(["country_code", "year"]).reset_index(drop=True)

    coverage = (
        panel.groupby("country_code", dropna=False)["co2_total_mmtco2"]
        .apply(lambda s: s.notna().sum())
        .reset_index(name="co2_non_null_years")
    )
    keep_codes = coverage.loc[coverage["co2_non_null_years"] >= MIN_NON_NULL_YEARS, "country_code"]

    panel = panel[panel["country_code"].isin(keep_codes)].copy()
    return panel.reset_index(drop=True)


def add_derived_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()

    df["population_people"] = df["population_thousands"] * 1_000
    df["gdp_usd_ppp"] = df["gdp_billion_ppp_usd"] * 1_000_000_000

    df["co2_per_capita_tonnes"] = safe_divide(df["co2_total_mmtco2"] * 1_000_000, df["population_people"])
    df["co2_per_gdp_tonnes_per_usd"] = safe_divide(df["co2_total_mmtco2"] * 1_000_000, df["gdp_usd_ppp"])
    df["gdp_per_capita_usd_ppp"] = safe_divide(df["gdp_usd_ppp"], df["population_people"])

    df["renewables_share"] = safe_divide(df["renewables_generation_bkwh"], df["electricity_generation_bkwh"])
    df["fossil_share"] = safe_divide(df["fossil_generation_bkwh"], df["electricity_generation_bkwh"])
    df["hydro_share"] = safe_divide(df["hydro_generation_bkwh"], df["electricity_generation_bkwh"])
    df["wind_share"] = safe_divide(df["wind_generation_bkwh"], df["electricity_generation_bkwh"])
    df["solar_share"] = safe_divide(df["solar_generation_bkwh"], df["electricity_generation_bkwh"])

    df = df.sort_values(["country_code", "year"]).copy()
    group = df.groupby("country_code", dropna=False)

    for col in [
        "co2_total_mmtco2",
        "gdp_billion_ppp_usd",
        "co2_per_capita_tonnes",
        "co2_per_gdp_tonnes_per_usd",
        "renewables_share",
        "fossil_share",
    ]:
        df[f"{col}_pct_change_1y"] = group[col].pct_change() * 100.0
        df[f"{col}_pct_change_3y"] = group[col].pct_change(periods=3) * 100.0
        df[f"{col}_pct_change_5y"] = group[col].pct_change(periods=5) * 100.0

    return df


def compute_country_slopes(panel: pd.DataFrame, value_col: str, years_back: int = 5) -> pd.DataFrame:
    records = []

    for (code, name), g in panel.groupby(["country_code", "country_name"], dropna=False):
        g = g.sort_values("year")
        g = g[g[value_col].notna()].tail(years_back)

        if len(g) < 3:
            slope = np.nan
        else:
            x = g["year"].to_numpy(dtype=float)
            y = g[value_col].to_numpy(dtype=float)
            slope = np.polyfit(x, y, 1)[0]

        records.append({
            "country_code": code,
            "country_name": name,
            f"slope_{value_col}": slope
        })

    return pd.DataFrame(records)


def build_country_summary(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[panel["year"] == YEAR_START].copy()
    cur = panel[panel["year"] == YEAR_END].copy()

    keep_cols = [
        "country_code", "country_name",
        "co2_total_mmtco2", "gdp_billion_ppp_usd", "population_people",
        "co2_per_capita_tonnes", "co2_per_gdp_tonnes_per_usd",
        "renewables_share", "fossil_share",
        "hydro_share", "wind_share", "solar_share",
        "gdp_per_capita_usd_ppp",
    ]

    base = base[keep_cols].rename(
        columns={c: f"{c}_2007" for c in keep_cols if c not in ["country_code", "country_name"]}
    )
    cur = cur[keep_cols].rename(
        columns={c: f"{c}_2023" for c in keep_cols if c not in ["country_code", "country_name"]}
    )

    summary = base.merge(cur, on="country_code", how="inner", suffixes=("", "_cur"))

    if "country_name_cur" in summary.columns:
        summary["country_name"] = summary["country_name"].fillna(summary["country_name_cur"])
        summary = summary.drop(columns=["country_name_cur"])

    summary["co2_change_2007_2023_pct"] = pct_change_from_base(
        summary["co2_total_mmtco2_2023"], summary["co2_total_mmtco2_2007"]
    )
    summary["gdp_change_2007_2023_pct"] = pct_change_from_base(
        summary["gdp_billion_ppp_usd_2023"], summary["gdp_billion_ppp_usd_2007"]
    )
    summary["co2_per_capita_change_2007_2023_pct"] = pct_change_from_base(
        summary["co2_per_capita_tonnes_2023"], summary["co2_per_capita_tonnes_2007"]
    )
    summary["co2_per_gdp_change_2007_2023_pct"] = pct_change_from_base(
        summary["co2_per_gdp_tonnes_per_usd_2023"], summary["co2_per_gdp_tonnes_per_usd_2007"]
    )
    summary["renewables_share_change_2007_2023_pct_points"] = (
        summary["renewables_share_2023"] - summary["renewables_share_2007"]
    ) * 100.0
    summary["fossil_share_change_2007_2023_pct_points"] = (
        summary["fossil_share_2023"] - summary["fossil_share_2007"]
    ) * 100.0

    summary["co2_vs_gdp_ratio_2007_2023"] = safe_divide(
        summary["co2_change_2007_2023_pct"],
        summary["gdp_change_2007_2023_pct"]
    )

    co2_slope = compute_country_slopes(panel, "co2_total_mmtco2", years_back=5)
    ren_slope = compute_country_slopes(panel, "renewables_share", years_back=5)

    summary = summary.merge(co2_slope, on=["country_code", "country_name"], how="left")
    summary = summary.merge(ren_slope, on=["country_code", "country_name"], how="left")

    essential = [
        "co2_total_mmtco2_2007",
        "co2_total_mmtco2_2023",
        "gdp_billion_ppp_usd_2007",
        "gdp_billion_ppp_usd_2023",
    ]
    summary = summary.dropna(subset=essential).copy()

    return summary.reset_index(drop=True)


# ============================================================
# CLUSTERING
# ============================================================

def label_cluster(row: pd.Series) -> str:
    dec = row["co2_vs_gdp_ratio_2007_2023"]
    renew = row["renewables_share_change_2007_2023_pct_points"]
    co2_change = row["co2_change_2007_2023_pct"]

    if pd.notna(dec) and dec < 0 and pd.notna(renew) and renew > 5:
        return "Decoupling leaders"
    if pd.notna(co2_change) and co2_change > 50 and pd.notna(renew) and renew < 5:
        return "Rapid emitters"
    if pd.notna(renew) and renew > 10 and pd.notna(co2_change) and co2_change <= 20:
        return "Transitioning systems"
    if pd.notna(co2_change) and co2_change < 0:
        return "Declining emitters"
    return "Mixed pathway"


def run_clustering(summary: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    out = summary.copy()

    if out.empty:
        raise ValueError("Summary dataframe is empty.")

    feature_cols = [
        "co2_total_mmtco2_2023",
        "co2_per_capita_tonnes_2023",
        "co2_per_gdp_tonnes_per_usd_2023",
        "renewables_share_2023",
        "fossil_share_2023",
        "co2_change_2007_2023_pct",
        "gdp_change_2007_2023_pct",
        "co2_per_gdp_change_2007_2023_pct",
        "renewables_share_change_2007_2023_pct_points",
        "slope_co2_total_mmtco2",
        "slope_renewables_share",
        "co2_vs_gdp_ratio_2007_2023",
    ]

    X = out[feature_cols].copy()
    X["co2_total_mmtco2_2023"] = np.log1p(X["co2_total_mmtco2_2023"])

    effective_n_clusters = min(n_clusters, len(out))
    if effective_n_clusters < 2:
        out["cluster_id"] = 0
        out["cluster_name"] = "Single available group"
        return out

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("cluster", KMeans(n_clusters=effective_n_clusters, random_state=42, n_init=20))
    ])

    out["cluster_id"] = pipe.fit_predict(X)

    profiles = (
        out.groupby("cluster_id", as_index=False)[[
            "co2_vs_gdp_ratio_2007_2023",
            "renewables_share_change_2007_2023_pct_points",
            "co2_change_2007_2023_pct"
        ]]
        .median()
    )
    profiles["cluster_name"] = profiles.apply(label_cluster, axis=1)

    out = out.merge(profiles[["cluster_id", "cluster_name"]], on="cluster_id", how="left")
    return out


def attach_clusters_to_panel(panel: pd.DataFrame, clustered_summary: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(
        clustered_summary[["country_code", "cluster_id", "cluster_name"]],
        on="country_code",
        how="left"
    )


# ============================================================
# PLOT
# ============================================================

def build_interactive_plot(panel_with_clusters: pd.DataFrame, save_path: Path) -> None:
    plot_df = panel_with_clusters[
        [
            "country_code",
            "country_name",
            "year",
            "cluster_name",
            "co2_total_mmtco2",
            "co2_per_capita_tonnes",
            "gdp_per_capita_usd_ppp",
            "renewables_share",
            "fossil_share",
            "co2_per_gdp_tonnes_per_usd",
        ]
    ].dropna(subset=["co2_total_mmtco2", "co2_per_capita_tonnes", "gdp_per_capita_usd_ppp"]).copy()

    plot_df["renewables_share_pct"] = plot_df["renewables_share"] * 100.0
    plot_df["fossil_share_pct"] = plot_df["fossil_share"] * 100.0

    fig = px.scatter(
        plot_df,
        x="gdp_per_capita_usd_ppp",
        y="co2_per_capita_tonnes",
        animation_frame="year",
        animation_group="country_code",
        size="co2_total_mmtco2",
        color="cluster_name",
        hover_name="country_name",
        hover_data={
            "country_code": True,
            "co2_total_mmtco2": ":.2f",
            "co2_per_capita_tonnes": ":.2f",
            "gdp_per_capita_usd_ppp": ":.0f",
            "renewables_share_pct": ":.1f",
            "fossil_share_pct": ":.1f",
            "co2_per_gdp_tonnes_per_usd": ":.8f",
            "year": True,
        },
        log_x=True,
        size_max=55,
        title="Carbon transition landscape, 2007–2023"
    )

    fig.update_layout(
        xaxis_title="GDP per capita (PPP USD, log scale)",
        yaxis_title="CO₂ per capita (tonnes)",
        legend_title="Transition cluster",
        template="plotly_white",
        height=760
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"Saved interactive plot to: {save_path.resolve()}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Building metric registry...")
    registry = build_metric_registry(BASE_DIR, CORE_FILES)

    print("\nBuilding base panel from selected metrics...")
    panel = build_base_panel(registry)
    print(f"Base panel shape: {panel.shape}")
    print(f"Countries in base panel: {panel['country_code'].nunique()}")
    print(panel[["country_code", "country_name"]].drop_duplicates().head(10))

    print("\nAdding derived features...")
    panel = add_derived_features(panel)

    print("\nBuilding country summary...")
    summary = build_country_summary(panel)
    print(f"Summary shape: {summary.shape}")
    print(f"Countries in summary: {summary['country_code'].nunique()}")
    print(summary[["country_code", "country_name"]].head(10))

    print("\nRunning clustering...")
    summary_clustered = run_clustering(summary, n_clusters=N_CLUSTERS)

    print("\nAttaching clusters back to panel...")
    panel_clustered = attach_clusters_to_panel(panel, summary_clustered)

    panel_csv = OUTPUT_DIR / "country_year_panel_2007_2023.csv"
    summary_csv = OUTPUT_DIR / "country_summary_2023_clusters.csv"
    panel_parquet = OUTPUT_DIR / "country_year_panel_2007_2023.parquet"
    summary_parquet = OUTPUT_DIR / "country_summary_2023_clusters.parquet"

    panel_clustered.to_csv(panel_csv, index=False)
    summary_clustered.to_csv(summary_csv, index=False)
    panel_clustered.to_parquet(panel_parquet, index=False)
    summary_clustered.to_parquet(summary_parquet, index=False)

    print(f"\nSaved panel csv: {panel_csv.resolve()}")
    print(f"Saved summary csv: {summary_csv.resolve()}")

    print("\nBuilding interactive plot...")
    build_interactive_plot(
        panel_clustered,
        OUTPUT_DIR / "carbon_transition_landscape_2007_2023.html"
    )

    print("\nDone.")

if __name__ == "__main__":
    main()