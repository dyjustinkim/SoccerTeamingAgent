import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def reverse_table(df):
    df2 = pd.DataFrame({
    "match id": df["match id"],
    "team1 name": df["team2 name"],
    "team2 name": df["team1 name"],
    "team1 goals": df["team2 goals"],
    "team2 goals": df["team1 goals"],
    "team1 xG": df["team2 xG"],
    "team2 xG": df["team1 xG"],
    "team1 formation": df["team2 formation"],
    "team2 formation": df["team1 formation"],
    "team1 poss": df["team2 poss"],
    "team2 poss": df["team1 poss"]
    })
    return df2


def expected_points1(row):
    if row["team1 goals"] == row["team2 goals"]:
        return 1
    elif row["team1 goals"] > row["team2 goals"]:
        return 3
    elif row["team1 goals"] < row["team2 goals"]:
        return 0
    
def clustering(df):
    features = ["team1 goals", "team2 goals", "team1 xG", "team2 xG"]
    X = df[features].copy()
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("Cluster centroids (standardized):")
    print(centroids)
    offensive_cluster = centroids["team1 xG"].idxmax()
    defensive_cluster = centroids["team1 xG"].idxmin()
    cluster_map = {offensive_cluster: "offensive", defensive_cluster: "defensive"}
    df["team1 tactics"] = df["cluster"].map(cluster_map)
    return df

def possession_clustering(row):
    if row["team1 tactics"] == "offensive" and int(row["team1 poss"].split("%")[0]) >= 50:
        return "Possession"
    elif row["team1 tactics"] == "offensive" and int(row["team1 poss"].split("%")[0]) < 50:
        return "Counter-Attack"
    elif row["team1 tactics"] == "defensive" and int(row["team1 poss"].split("%")[0]) >= 50:
        return "High Press"
    elif row["team1 tactics"] == "defensive" and int(row["team1 poss"].split("%")[0]) < 50:
        return "Low Block"


def cluster_historical(file_name, output_name):
    df = pd.read_csv(file_name)
    df["match id"] = df.index
    df2 = reverse_table(df)
    df = pd.concat([df, df2], ignore_index=True)
    clustered_df = clustering(df)
    clustered_df["team1 points"] = clustered_df.apply(expected_points1, axis=1)
    clustered_df["team1 strat"] = clustered_df.apply(possession_clustering, axis=1)

    
    clustered_df.to_csv(f"evaluation/{output_name}", index=False)


cluster_historical("evaluation/Prem_22-23.csv", "Prem_22-23_clustered.csv")
