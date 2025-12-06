import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
import time


t0 = time.perf_counter() 
logs = pd.read_csv("tottenham_match_cleaned.csv")
centroids = pd.read_csv("player_clusters_by_position.csv")

t_load0 = time.perf_counter()
t_load1 = time.perf_counter()
# rename_map = {
#     "Passes_Cmp": "AccLB",
#     "Passes_Att": "InAccLB",
#     "Passes_PrgP": "AccSP",
#     "Performance_Crs": "InAccSP",
#     "Take-Ons_Att": "InAccCr",
#     "Performance_Fld": "AccFrK",
#     "Carries_Carries": "Total Dribbles",
#     "Min": "KG"
# }
# logs.rename(columns=rename_map, inplace=True)

segmented_roles = []
def map_pos(pos):
    pos = pos.upper()
    if "GK" in pos: return "GK"
    if "CB" in pos or "D(C" in pos: return "CB"
    if "CM" in pos or "DM" in pos or "M(" in pos or "DMC" in pos: return "CMF"
    if "WB" in pos or "D(LR" in pos: return "WB"
    if "LW" in pos or "RW" in pos or "AM" in pos: return "Wing"
    if "FW" in pos or "STRIKER" in pos: return "FW"
    return None


t_cluster0 = time.perf_counter()
for pos in centroids["Target"].unique():
    pos_centroids = centroids[centroids["Target"] == pos].drop(columns=["Target", "Cluster", "SegmentedPosition"])
    logs["MappedPos"] = logs["Pos"].apply(map_pos)
    pos_players = logs[logs["MappedPos"] == pos]    

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
t_cluster1 = time.perf_counter()

# 3. Save result
t_save0 = time.perf_counter()
segmented_df.to_csv("player_match_with_segments2.csv", index=False)
t_save1 = time.perf_counter()

t1 = time.perf_counter()  # end total timing

print("Saved → player_match_with_segments2.csv")

print("\nTiming (seconds):")
print(f"load_csvs          : {t_load1 - t_load0:.6f}")
print(f"cluster_assignment : {t_cluster1 - t_cluster0:.6f}")
print(f"save_csv           : {t_save1 - t_save0:.6f}")
print(f"total_pipeline     : {t1 - t0:.6f}")