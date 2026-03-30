#!/usr/bin/env python3
"""
Download & organize EIA international energy data into wide-format CSVs.
Uses only Python stdlib. Downloads INTL.zip from EIA bulk data endpoint,
parses the JSON-lines file, uses the embedded category hierarchy to classify
series, groups by metric (geoset), and pivots each group into a wide CSV
(countries as rows, years as columns).
"""

import csv
import io
import json
import os
import re
import urllib.request
import zipfile
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_URL = "https://www.eia.gov/opendata/bulk/INTL.zip"
ZIP_PATH = os.path.join(BASE_DIR, "INTL.zip")


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def download_zip():
    if os.path.exists(ZIP_PATH):
        print(f"  Using cached {ZIP_PATH}")
        return
    print(f"  Downloading {ZIP_URL} ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"  Saved to {ZIP_PATH}")


# ---------------------------------------------------------------------------
# Top-level folder mapping: map ancestor category names to our folder names
# ---------------------------------------------------------------------------
ANCESTOR_TO_FOLDER = {
    "Petroleum and other liquids": "petroleum-and-other-liquids",
    "Total petroleum and other liquids": "petroleum-and-other-liquids",
    "Crude oil including lease condensate": "petroleum-and-other-liquids",
    "Crude oil, NGPL, and other liquids": "petroleum-and-other-liquids",
    "Refined petroleum products": "petroleum-and-other-liquids",
    "Motor gasoline": "petroleum-and-other-liquids",
    "Jet fuel": "petroleum-and-other-liquids",
    "Kerosene": "petroleum-and-other-liquids",
    "Distillate fuel oil": "petroleum-and-other-liquids",
    "Residual fuel oil": "petroleum-and-other-liquids",
    "Liquefied Petroleum Gases": "petroleum-and-other-liquids",
    "Liquefied petroleum gases and ethane": "petroleum-and-other-liquids",
    "Liquefied petroleum gases and ethane (refinery)": "petroleum-and-other-liquids",
    "Liquefied petroleum gases (nonrefinery)": "petroleum-and-other-liquids",
    "Other petroleum liquids": "petroleum-and-other-liquids",
    "Other liquids": "petroleum-and-other-liquids",
    "NGPL": "petroleum-and-other-liquids",
    "Refinery processing gain": "petroleum-and-other-liquids",
    "LPG": "petroleum-and-other-liquids",
    "Ethane": "petroleum-and-other-liquids",
    "Ethane (non refinery)": "petroleum-and-other-liquids",
    "Oil": "petroleum-and-other-liquids",
    "OECD imports": "petroleum-and-other-liquids",
    "OECD Europe imports, by source": "petroleum-and-other-liquids",
    "France imports, by source": "petroleum-and-other-liquids",
    "Germany imports, by source": "petroleum-and-other-liquids",
    "Italy imports, by source": "petroleum-and-other-liquids",
    "Japan imports, by source": "petroleum-and-other-liquids",
    "South Korea imports, by source": "petroleum-and-other-liquids",
    "U.S. imports, by source": "petroleum-and-other-liquids",
    "Canada imports, by source": "petroleum-and-other-liquids",
    "United Kingdom imports, by source": "petroleum-and-other-liquids",

    "Dry natural gas": "natural-gas",
    "Natural gas": "natural-gas",
    "Gross natural gas": "natural-gas",
    "Consumed natural gas": "natural-gas",
    "Vented and flared natural gas": "natural-gas",
    "Reinjected natural gas": "natural-gas",

    "Coal": "coal-and-coke",
    "Coal and coke": "coal-and-coke",
    "Anthracite": "coal-and-coke",
    "Bituminous": "coal-and-coke",
    "Subbituminous": "coal-and-coke",
    "Lignite": "coal-and-coke",
    "Metallurgical coke": "coal-and-coke",
    "Metallurgical coal": "coal-and-coke",

    "Electricity": "electricity",
    "Nuclear": "electricity",
    "Fossil fuels": "electricity",
    "Renewables": "electricity",
    "Hydroelectricity": "electricity",
    "Hydroelectric pumped storage": "electricity",
    "Non-hydroelectric renewables": "electricity",
    "Geothermal": "electricity",
    "Solar": "electricity",
    "Solar, tide, wave, fuel cell": "electricity",
    "Tide and wave": "electricity",
    "Wind": "electricity",
    "Biomass and waste": "electricity",
    "Other gases": "electricity",

    "Biofuels": "biofuels",
    "Fuel ethanol": "biofuels",
    "Biomass-based diesel": "biofuels",

    "Primary energy": "total-energy",

    "Energy intensity": "energy-intensity",
    "Population": "energy-intensity",
    "Gross domestic product": "energy-intensity",

    "CO2 emissions": "co2-emissions",

    "Nuclear, renewables, and other": "total-energy",
    "Renewables and other": "total-energy",
}


def parse_zip(zip_path):
    """
    Two-pass parse of the INTL.txt JSON-lines file:
    Pass 1: Build category hierarchy and series→category mapping
    Pass 2: Yield annual series objects with their resolved folder assignment
    """
    # --- Pass 1: categories ---
    print("  Pass 1: Building category hierarchy ...")
    categories = {}  # cid → {name, parent, childseries_prefixes}
    # Map geoset_prefix (product-activity) → list of ancestor names
    geoset_to_ancestors = {}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        fname = [n for n in names if n.endswith(".txt") or n.endswith(".json")]
        fname = fname[0] if fname else names[0]

        with zf.open(fname) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "category_id" in obj and "series_id" not in obj:
                    cid = obj["category_id"]
                    categories[cid] = {
                        "name": obj.get("name", ""),
                        "parent": obj.get("parent_category_id", ""),
                        "childseries": obj.get("childseries", []),
                    }

    # Build ancestry for categories with series
    def get_ancestors(cid, depth=0):
        if depth > 10 or cid not in categories:
            return []
        cat = categories[cid]
        chain = [cat["name"]]
        if cat["parent"] and cat["parent"] in categories:
            chain = get_ancestors(cat["parent"], depth + 1) + chain
        return chain

    for cid, cat in categories.items():
        if cat["childseries"]:
            ancestors = get_ancestors(cid)
            # Extract geoset prefix from first child series
            sample = cat["childseries"][0]
            parts = sample.replace("INTL.", "").split("-")
            if len(parts) >= 2:
                prefix = f"{parts[0]}-{parts[1]}"
                geoset_to_ancestors[prefix] = ancestors

    # Build geoset_prefix → folder mapping
    geoset_to_folder = {}
    for prefix, ancestors in geoset_to_ancestors.items():
        folder = None
        for anc in ancestors:
            if anc in ANCESTOR_TO_FOLDER:
                folder = ANCESTOR_TO_FOLDER[anc]
                break
        geoset_to_folder[prefix] = folder or "other"

    # Also store a nice label: activity name from ancestry
    geoset_to_label = {}
    for prefix, ancestors in geoset_to_ancestors.items():
        # ancestors is like ["Coal", "Production"] or ["Electricity", "Nuclear", "Generation"]
        # Use the full path as the label
        geoset_to_label[prefix] = " - ".join(ancestors)

    print(f"    Found {len(categories)} categories, {len(geoset_to_folder)} geoset prefixes")

    # --- Pass 2: yield series with folder ---
    print("  Pass 2: Reading series ...")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(fname) as f:
            for line_num, line in enumerate(io.TextIOWrapper(f, encoding="utf-8"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "series_id" not in obj or obj.get("f") != "A":
                    continue

                sid = obj["series_id"]
                core = sid.replace("INTL.", "").replace(".A", "")
                parts = core.split("-")
                if len(parts) < 3:
                    continue

                prefix = f"{parts[0]}-{parts[1]}"
                folder = geoset_to_folder.get(prefix, "other")
                label = geoset_to_label.get(prefix, "")

                obj["_folder"] = folder
                obj["_label"] = label
                yield obj

                if line_num % 100000 == 0:
                    print(f"    ... processed {line_num} lines")


def build_geoset_key(series):
    """
    Group key = geoset_id (product-activity-unit, without region).
    Falls back to constructing from series_id.
    """
    geoset = series.get("geoset_id", "")
    if geoset:
        return geoset
    sid = series["series_id"]
    core = sid.replace("INTL.", "").replace(".A", "")
    parts = core.split("-")
    if len(parts) >= 4:
        return f"INTL.{parts[0]}-{parts[1]}-{parts[3]}.A"
    return sid


def extract_region_code(series):
    core = series["series_id"].replace("INTL.", "").replace(".A", "")
    parts = core.split("-")
    return parts[2] if len(parts) >= 3 else None


def extract_country_name(series):
    """Extract country/region name from the series name field.
    Format: 'Metric description, Country/Region, Annual'
    """
    name = series.get("name", "")
    parts = [p.strip() for p in name.split(",")]
    # Remove "Annual" or "Monthly" at the end
    if parts and parts[-1].strip().lower() in ("annual", "monthly", "quarterly"):
        parts = parts[:-1]
    # The country/region is the last remaining part
    if len(parts) >= 2:
        return parts[-1].strip()
    return extract_region_code(series) or "Unknown"


def build_filename(label, units):
    """Build a descriptive filename from the category label and units."""
    # label is like "Coal - Production" or "Electricity - Nuclear - Generation"
    # units is like "billion kilowatthours" or "1000 metric tons"
    parts = [p.strip() for p in label.split(" - ")] if label else []

    # Build name from category path
    if len(parts) >= 2:
        # e.g. "coal-production" or "electricity-nuclear-generation"
        name = "-".join(parts)
    elif parts:
        name = parts[0]
    else:
        return None

    # Add unit info for disambiguation (shortened)
    unit_slug = slugify(units) if units else ""
    # Shorten common unit patterns
    unit_short = unit_slug
    for long, short in [
        ("billion-kilowatthours", "bkwh"),
        ("million-kilowatthours", "mkwh"),
        ("thousand-barrels-per-day", "tbpd"),
        ("thousand-barrels", "tbbl"),
        ("million-barrels-per-day", "mbpd"),
        ("million-barrels", "mbbl"),
        ("billion-cubic-feet", "bcf"),
        ("trillion-cubic-feet", "tcf"),
        ("1000-metric-tons", "kmt"),
        ("million-metric-tons", "mmt"),
        ("million-metric-tonnes-carbon-dioxide", "mmtco2"),
        ("quadrillion-btu", "qbtu"),
        ("terajoules", "tj"),
        ("million-metric-tons-of-oil-equivalent", "mtoe"),
        ("thousand-megawatts", "tmw"),
        ("thousand-megawatthours", "tmwh"),
        ("million-kilowatts", "mk"),
        ("megawatts", "mw"),
        ("mmbtu-per-person", "mbtupp"),
        ("1000-btu-per-2015-gdp-ppp", "btugdp"),
        ("millions", "m"),
        ("2015-us-dollars-per-million-btu", "usdpmbtu"),
    ]:
        if unit_slug == long:
            unit_short = short
            break

    slug = slugify(name)
    if unit_short:
        slug = f"{slug}-{unit_short}"

    if len(slug) > 80:
        slug = slug[:80].rstrip("-")
    return slug


def main():
    print("Step 1: Download INTL.zip")
    download_zip()

    # Clean up previous output
    for folder in [
        "petroleum-and-other-liquids", "natural-gas", "coal-and-coke",
        "electricity", "hydrocarbon-gas-liquids", "biofuels",
        "total-energy", "energy-intensity", "co2-emissions", "other",
    ]:
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(".csv"):
                    os.remove(os.path.join(folder_path, f))

    print("\nStep 2: Parse and group series by geoset")
    # Group: geoset_key → {folder, label, units, regions: {code: {name, data}}}
    metrics = defaultdict(lambda: {
        "folder": None, "label": None, "units": None, "regions": {}
    })
    processed = 0

    for series in parse_zip(ZIP_PATH):
        gkey = build_geoset_key(series)
        region_code = extract_region_code(series)
        if not region_code:
            continue

        data_points = series.get("data", [])
        if not data_points:
            continue

        country_name = extract_country_name(series)

        m = metrics[gkey]
        if m["folder"] is None:
            m["folder"] = series["_folder"]
            m["label"] = series["_label"]
            m["units"] = series.get("units", "")
        m["regions"][region_code] = {
            "name": country_name,
            "data": {str(yr): val for yr, val in data_points},
        }
        processed += 1

    print(f"  Processed {processed} series into {len(metrics)} geosets")

    print("\nStep 3: Write wide-format CSVs")
    readme_rows = []
    used_paths = set()

    for gkey, info in sorted(metrics.items()):
        folder = info["folder"]
        regions = info["regions"]
        if not regions:
            continue

        all_years = set()
        for r in regions.values():
            all_years.update(r["data"].keys())
        all_years = sorted(y for y in all_years if y.isdigit() and 1900 <= int(y) <= 2030)
        if not all_years:
            continue

        file_slug = build_filename(info["label"], info["units"])
        if not file_slug:
            file_slug = slugify(gkey)
        if not file_slug:
            continue

        folder_path = os.path.join(BASE_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)

        csv_filename = f"{file_slug}.csv"
        csv_path = os.path.join(folder_path, csv_filename)

        counter = 1
        while csv_path in used_paths:
            csv_filename = f"{file_slug}-{counter}.csv"
            csv_path = os.path.join(folder_path, csv_filename)
            counter += 1
        used_paths.add(csv_path)

        header = ["Country", "Code"] + all_years

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for code in sorted(regions.keys()):
                r = regions[code]
                row = [r["name"], code]
                for yr in all_years:
                    val = r["data"].get(yr, "")
                    row.append(val)
                writer.writerow(row)

        rel_path = os.path.join(folder, csv_filename)
        # Build a human-readable description
        desc = info["label"] or gkey
        if info["units"]:
            desc = f"{desc} ({info['units']})"

        readme_rows.append({
            "file": rel_path,
            "description": desc,
            "units": info.get("units", ""),
            "rows": len(regions),
        })

    print(f"  Wrote {len(readme_rows)} CSV files")

    print("\nStep 4: Write README.csv index")
    readme_path = os.path.join(BASE_DIR, "README.csv")
    with open(readme_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Description", "Units", "Row Count"])
        for row in sorted(readme_rows, key=lambda r: r["file"]):
            writer.writerow([row["file"], row["description"], row["units"], row["rows"]])

    print(f"  Wrote {readme_path}")

    # Summary by folder
    folder_counts = defaultdict(int)
    for row in readme_rows:
        folder_counts[row["file"].split("/")[0]] += 1
    print("\n  Files per folder:")
    for folder, count in sorted(folder_counts.items()):
        print(f"    {folder}: {count}")

    print("\nDone! Check README.csv for a full index of available data.")


if __name__ == "__main__":
    main()
