import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("player_match_with_segments.csv")

df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace("\xa0", "")
    .str.lower()
)
df = df.loc[:, ~df.columns.duplicated()]


if not {"matchdate", "opponent", "segmentedposition"}.issubset(df.columns):
    raise KeyError("missing one of matchdate/opponent/segmentedposition")

numeric_cols = df.select_dtypes(include=["number"]).columns.drop(["#"], errors="ignore")


team_vectors = df.groupby(["matchdate", "opponent", "segmentedposition"])[numeric_cols].mean()


team_vectors = team_vectors.unstack(level="segmentedposition")
team_vectors.columns = [f"{pos}_{stat}" for stat, pos in team_vectors.columns]


team_vectors = team_vectors.reset_index(level=["matchdate", "opponent"])


team_vectors = team_vectors.loc[:, ~team_vectors.columns.duplicated()]


# merge team vector with match result data 
meta = pd.read_csv("match_data/tottenham_match_results.csv")
meta["Opponent"] = meta["Opponent"].astype(str).str.strip().str.lower()
team_vectors["opponent"] = team_vectors["opponent"].astype(str).str.strip().str.lower()
df = pd.merge(team_vectors, meta, left_on="opponent", right_on="Opponent", how="inner") # merge by opponent

# drop redundant columns from meta
drop_cols = [
    "Date", "Opponent", "Match Report", "Notes",
    "Venue", "Day", "Round", "Time", "Attendance",
    "Referee", "Captain", "Opp Formation", "matchdate"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# create formation label
le_formation = LabelEncoder()
df["FormationLabel"] = le_formation.fit_transform(df["Formation"].astype(str))



# create tactical style feature
features = ["GF", "GA", "xG", "xGA"]

# normalize numeric features
scaler = StandardScaler()
X = scaler.fit_transform(df[features].fillna(0))

# use kmeans clustering (2 clusters: offensive / defensive)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["TacticalStyleCluster"] = kmeans.fit_predict(X)

# summarize clusters
cluster_summary = df.groupby("TacticalStyleCluster")[features].mean()

# offensiveness metric: total goal activity (sum of gf, xg, ga, xga)
cluster_summary["Offensiveness"] = (cluster_summary["GF"] + cluster_summary["GA"] + cluster_summary["xG"] + cluster_summary["xGA"])

# assign cluster with higher offensiveness to be offensive
offensive_cluster = cluster_summary["Offensiveness"].idxmax()
df["TacticalStyle"] = df["TacticalStyleCluster"].apply(lambda c: "Offensive" if c == offensive_cluster else "Defensive")

# create tactical style label
le_tacticalstyle = LabelEncoder()
df["TacticalStyleLabel"] = le_tacticalstyle.fit_transform(df["TacticalStyle"])



# create strategy feature
df["StrategyCluster"] = None
df["Strategy"] = None

# Offensive: Possession vs Pressing
offensive_df = df[df["TacticalStyle"] == "Offensive"].copy()
if not offensive_df.empty:
    features_off = ["Poss", "xG", "xGA"]
    X_off = StandardScaler().fit_transform(offensive_df[features_off].fillna(0))
    kmeans_off = KMeans(n_clusters=2, random_state=42, n_init=10)
    offensive_df["StrategyCluster"] = kmeans_off.fit_predict(X_off)

    summary_off = offensive_df.groupby("StrategyCluster")[features_off].mean()
    possession_cluster = summary_off["Poss"].idxmax() # use cluster with higher posession
    offensive_df["Strategy"] = offensive_df["StrategyCluster"].apply(lambda c: "Possession" if c == possession_cluster else "Counter-attack")

# Defensive: Low-block vs High-press
defensive_df = df[df["TacticalStyle"] == "Defensive"].copy()
if not defensive_df.empty:
    features_def = ["Poss", "xG", "xGA"]
    X_def = StandardScaler().fit_transform(defensive_df[features_def].fillna(0))
    kmeans_def = KMeans(n_clusters=2, random_state=42, n_init=10)
    defensive_df["StrategyCluster"] = kmeans_def.fit_predict(X_def)

    summary_def = defensive_df.groupby("StrategyCluster")[features_def].mean()
    highpress_cluster = summary_def["Poss"].idxmax() # use cluster with higher possession
    defensive_df["Strategy"] = defensive_df["StrategyCluster"].apply(lambda c: "High-press" if c == highpress_cluster else "Low-block")

# Combine back
df = pd.concat([offensive_df, defensive_df]).sort_index()

# create strategy label
le_strategy = LabelEncoder()
df["StrategyLabel"] = le_strategy.fit_transform(df["Strategy"])

print("\n=== Strategy Summary ===")
print(df[["opponent", "TacticalStyle", "Strategy"]])




# merge with league data
league = pd.read_csv("match_data/league_table.csv")
league["Squad"] = league["Squad"].astype(str).str.strip().str.lower()
merged = pd.merge(df, league, left_on="opponent", right_on="Squad", how="left", suffixes=("_team", "_league"))

# export final input vector
merged.to_csv("input_vectors.csv", index=False)

print("Saved input_vectors.csv", df.shape)
