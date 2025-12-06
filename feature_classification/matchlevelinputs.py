import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

def matchlevelinputs(logs, centroids, output_file):

    logs = pd.read_csv(logs)
    centroids = pd.read_csv(centroids)

    rename_map = {
        "Passes_Cmp": "AccLB",
        "Passes_Att": "InAccLB",
        "Passes_PrgP": "AccSP",
        "Performance_Crs": "InAccSP",
        "Take-Ons_Att": "InAccCr",
        "Performance_Fld": "AccFrK",
        "Carries_Carries": "Total Dribbles",
        "Min": "KG"
    }
    logs.rename(columns=rename_map, inplace=True)

    segmented_roles = []

    for pos in centroids["Target"].unique():
        pos_centroids = centroids[centroids["Target"] == pos].drop(columns=["Target", "Cluster", "SegmentedPosition"])
        pos_players = logs[logs["Pos"].str.contains(pos, case=False, na=False)].copy()

        if pos_players.empty:
            continue

        features = [c for c in pos_centroids.columns if c in pos_players.columns]
        if not features:
            print(f"No matching features for {pos}. Skipping.")
            continue

        X = pos_players[features].fillna(0).values
        C = pos_centroids[features].fillna(0).values

        labels, _ = pairwise_distances_argmin_min(X, C)
        pos_players["SegmentedPosition"] = [f"{pos}{l}" for l in labels]
        segmented_roles.append(pos_players)

    segmented_df = pd.concat(segmented_roles, ignore_index=True)
    segmented_df.to_csv(output_file, index=False)
    print("Saved → player_match_with_segments.csv")
