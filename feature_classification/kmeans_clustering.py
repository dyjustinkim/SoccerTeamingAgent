import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("boruta_confirmed_features.csv")
df = df.fillna(0)  

X = df.drop(columns=["Target"])
y = df["Target"]

positions = y.unique()
print(f"Positions found: {positions}\n")

all_results = []
for position in positions:
    print(f"\n{'='*60}")
    print(f"Processing position: {position}")
    print(f"{'='*60}")

    position_mask = y == position
    X_position = X[position_mask]
    
    if len(X_position) < 2:
        print(f"⚠️ Only {len(X_position)} player(s) in {position}, skipping clustering")
        continue
    
    print(f"Number of players: {len(X_position)}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_position)
    
    # Elbow method to determine optimal k
    inertias = []
    K_range = range(1, min(6, len(X_position)))  # Test k from 1 to 5 (or less if fewer players)
    
    for k in K_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_test.fit(X_scaled)
        inertias.append(kmeans_test.inertia_)
    
    # Apply k means, with k = 2. Creates segmented positions
    optimal_k = 2 if len(X_position) >= 2 else 1
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    position_results = df[position_mask].copy()
    position_results['Cluster'] = clusters
    position_results['SegmentedPosition'] = position + position_results['Cluster'].astype(str)
    
    all_results.append(position_results)
    
    #Cluster characteristics
    print(f"\nCluster Summary for {position}:")
    cluster_means = position_results.groupby('Cluster')[X.columns].mean()
    print(cluster_means.round(2))
    
    print(f"\nPlayers in each cluster:")
    for cluster_id in range(optimal_k):
        cluster_players = position_results[position_results['Cluster'] == cluster_id]
        print(f"  Cluster {cluster_id} ({position}{cluster_id}): {len(cluster_players)} players")
    

if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("player_clusters_by_position.csv", index=False)
    print(f"\n{'='*60}")
    print("Saved all clustered players to player_clusters_by_position.csv")
    print(f"{'='*60}")
    
    #Summary statistics 
    print("\n SUMMARY: Segmented Positions Distribution")
    print(final_df['SegmentedPosition'].value_counts().sort_index())
    
    print("\n Feature Averages by Segmented Position:")
    summary = final_df.groupby('SegmentedPosition')[X.columns].mean()
    print(summary.round(2))
else:
    print("No clustering results generated")