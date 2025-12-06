import pandas as pd
import time


t0 = time.perf_counter() 
# 1. Load your raw per-match stats (the one with duplicates)
df_match = pd.read_csv("match_stats_fbref.csv")

# 2. Load your big roster table (the one with all Tottenham players)
df_roster = pd.read_csv("Tottenham_Players.csv")

# Normalize names to compare safely
df_match["Player_clean"] = df_match["Player"].str.replace('"', '').str.strip().str.lower()
df_roster["Player_clean"] = df_roster["Player"].str.replace('"', '').str.strip().str.lower()

# 3. Keep ONLY players who are in the Tottenham roster
df_filtered = df_match[df_match["Player_clean"].isin(df_roster["Player_clean"])]

# 4. Drop duplicated rows
df_filtered = df_filtered.drop_duplicates(subset=["Player", "Pos", "KG"])

# 5. Output only the required columns
df_final = df_filtered[
    ["Player", "Pos", "AccLB", "InAccLB", "AccSP", "InAccSP",
     "InAccCr", "AccFrK", "Total Dribbles", "KG"]
]

df_final.to_csv("tottenham_match_cleaned.csv", index=False)
print(df_final)


t1 = time.perf_counter()  # END total timing

print(df_final)
print(f"\nTotal runtime (seconds): {t1 - t0:.6f}")