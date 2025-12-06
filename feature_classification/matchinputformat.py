import pandas as pd
import glob
import os

def matchinputformat(input_file, output_file):
    files = glob.glob(os.path.join(input_file, "*.csv"))

    all_matches = []

    for f in files:
        df = pd.read_csv(f)
        base = os.path.basename(f)


        parts = base.replace(".csv", "").split("_")
        date = parts[1] if len(parts) > 1 else "UnknownDate"
        opponent = parts[2] if len(parts) > 2 else "UnknownOpponent"

        df["MatchDate"] = date
        df["Opponent"] = opponent
        all_matches.append(df)


    merged = pd.concat(all_matches, ignore_index=True)


    merged = merged[merged["Player"].str.contains("Players") == False]
    merged.to_csv(output_file, index=False)
    print("Saved merged dataset:", merged.shape)
