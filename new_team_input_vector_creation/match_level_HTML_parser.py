# scraperv2.py

import pandas as pd
from pathlib import Path
import time

# Path to the HTML file you saved from FBref
HTML_FILE = Path(
    "Tottenham Hotspur vs. Wolverhampton Wanderers Match Report – Sunday December 29, 2024 _ FBref.com.html"
)

OUTPUT_CSV = "match_stats_fbref.csv"


def read_tables(html_path: Path):
    """
    Read all tables from the saved FBref HTML.
    """
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    # header=[0,1] lets pandas build the MultiIndex from the two header rows
    tables = pd.read_html(html_path, header=[0, 1])
    return tables


def find_level1(df: pd.DataFrame, level1_name: str):
    """
    In a MultiIndex columns frame, find the column whose 2nd level == level1_name.
    """
    for col in df.columns:
        if isinstance(col, tuple) and len(col) > 1 and col[1] == level1_name:
            return col
    raise KeyError(f"Column with level-1 name '{level1_name}' not found")


def extract_passing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a 'passing totals' table, pull out Player/Pos/Min and
    Total/Long Cmp/Att.
    """
    player_col = find_level1(df, "Player")
    pos_col = find_level1(df, "Pos")
    min_col = find_level1(df, "Min")

    out = df[
        [
            player_col,
            pos_col,
            min_col,
            ("Long", "Cmp"),
            ("Long", "Att"),
            ("Total", "Cmp"),
            ("Total", "Att"),
        ]
    ].copy()

    out.columns = [
        "Player",
        "Pos",
        "Min",
        "Long_Cmp",
        "Long_Att",
        "Total_Cmp",
        "Total_Att",
    ]

    # Drop aggregate row like "15 Players"
    out = out[out["Player"].astype(str).str.contains("Players") == False]

    out["Min"] = pd.to_numeric(out["Min"], errors="coerce")
    out["Long_Cmp"] = pd.to_numeric(out["Long_Cmp"], errors="coerce").fillna(0)
    out["Long_Att"] = pd.to_numeric(out["Long_Att"], errors="coerce").fillna(0)
    out["Total_Cmp"] = pd.to_numeric(out["Total_Cmp"], errors="coerce").fillna(0)
    out["Total_Att"] = pd.to_numeric(out["Total_Att"], errors="coerce").fillna(0)

    out = out.dropna(subset=["Min"])
    return out


def extract_pass_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From 'pass types' table, pull FK (free-kick passes) and Crs (crosses).
    """
    player_col = find_level1(df, "Player")
    pos_col = find_level1(df, "Pos")

    out = df[
        [
            player_col,
            pos_col,
            ("Pass Types", "FK"),
            ("Pass Types", "Crs"),
        ]
    ].copy()

    out.columns = ["Player", "Pos", "FK", "Crs"]
    out = out[out["Player"].astype(str).str.contains("Players") == False]

    out["FK"] = pd.to_numeric(out["FK"], errors="coerce").fillna(0)
    out["Crs"] = pd.to_numeric(out["Crs"], errors="coerce").fillna(0)

    return out


def extract_touches_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From touches/take-ons/carries table, pull minutes and take-on attempts.
    """
    player_col = find_level1(df, "Player")
    pos_col = find_level1(df, "Pos")
    min_col = find_level1(df, "Min")

    out = df[
        [
            player_col,
            pos_col,
            min_col,
            ("Take-Ons", "Att"),
        ]
    ].copy()

    out.columns = ["Player", "Pos", "Min", "TakeOns_Att"]
    out = out[out["Player"].astype(str).str.contains("Players") == False]

    out["Min"] = pd.to_numeric(out["Min"], errors="coerce")
    out["TakeOns_Att"] = (
        pd.to_numeric(out["TakeOns_Att"], errors="coerce").fillna(0)
    )

    out = out.dropna(subset=["Min"])
    return out


def scrape_match_stats_from_html(html_path: Path) -> pd.DataFrame:
    tables = read_tables(html_path)

    # Collect the three types of tables for BOTH teams
    passing_dfs = []
    pass_type_dfs = []
    touches_dfs = []

    for df in tables:
        if not isinstance(df.columns, pd.MultiIndex):
            continue

        cols0 = set(c[0] for c in df.columns if isinstance(c, tuple))

        # Passing totals tables have 'Total' and 'Long' with Cmp/Att
        if ("Total", "Cmp") in df.columns and ("Long", "Cmp") in df.columns:
            passing_dfs.append(extract_passing_features(df))

        # Pass types tables have top-level 'Pass Types'
        if ("Pass Types", "FK") in df.columns and ("Pass Types", "Crs") in df.columns:
            pass_type_dfs.append(extract_pass_type_features(df))

        # Touches tables have 'Take-Ons' top level with 'Att'
        if ("Take-Ons", "Att") in df.columns and ("Carries", "Carries") in df.columns:
            touches_dfs.append(extract_touches_features(df))

    if not passing_dfs or not pass_type_dfs or not touches_dfs:
        raise RuntimeError("Could not find all required tables in HTML.")

    passing_all = pd.concat(passing_dfs, ignore_index=True)
    types_all = pd.concat(pass_type_dfs, ignore_index=True)
    touches_all = pd.concat(touches_dfs, ignore_index=True)

    # Merge on Player name (unique per match)
    df = passing_all.merge(
        types_all[["Player", "FK", "Crs"]], on="Player", how="left"
    )
    df = df.merge(
        touches_all[["Player", "TakeOns_Att", "Min"]],
        on="Player",
        how="left",
        suffixes=("", "_touch"),
    )

    # If Min from touches is available, prefer that
    df["KG"] = df["Min_touch"].fillna(df["Min"])
    df["KG"] = pd.to_numeric(df["KG"], errors="coerce")

    # Compute your features
    df["AccLB"] = df["Long_Cmp"]
    df["InAccLB"] = df["Long_Att"] - df["Long_Cmp"]

    df["AccSP"] = df["Total_Cmp"]
    df["InAccSP"] = df["Total_Att"] - df["Total_Cmp"]

    df["InAccCr"] = df["Crs"]
    df["AccFrK"] = df["FK"]
    df["Total Dribbles"] = df["TakeOns_Att"]

    # Clean up and keep only the columns you care about
    df_out = df[
        [
            "Player",
            "Pos",
            "AccLB",
            "InAccLB",
            "AccSP",
            "InAccSP",
            "InAccCr",
            "AccFrK",
            "Total Dribbles",
            "KG",
        ]
    ].sort_values(["Pos", "Player"])

    return df_out.reset_index(drop=True)


if __name__ == "__main__":
    df_stats = scrape_match_stats_from_html(HTML_FILE)
    print(df_stats)
    df_stats.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")

    t0 = time.perf_counter()  # start total timer

    df_stats = scrape_match_stats_from_html(HTML_FILE)
    df_stats.to_csv(OUTPUT_CSV, index=False)

    t1 = time.perf_counter()  # end total timer

    print(df_stats)
    print(f"Saved → {OUTPUT_CSV}")
    print(f"Total runtime (s): {t1 - t0:.6f}")
