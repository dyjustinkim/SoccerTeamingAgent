import pandas as pd

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

numeric_cols = df.select_dtypes(include=["number"]).columns


team_vectors = df.groupby(["matchdate", "opponent", "segmentedposition"])[numeric_cols].mean()


team_vectors = team_vectors.unstack(level="segmentedposition")
team_vectors.columns = [f"{pos}_{stat}" for stat, pos in team_vectors.columns]


team_vectors = team_vectors.reset_index(level=["matchdate", "opponent"])


team_vectors = team_vectors.loc[:, ~team_vectors.columns.duplicated()]


team_vectors.to_csv("team_match_vectors.csv", index=False)
print("Saved team_match_vectors.csv", team_vectors.shape)
