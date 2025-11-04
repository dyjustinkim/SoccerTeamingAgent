import pandas as pd

url = "https://fbref.com/en/matches/8fa951f9/Tottenham-Hotspur-Aston-Villa-November-3-2024-Premier-League"
tables = pd.read_html(url)    # reads every table on the page
print(len(tables))            # number of tables found

# usually the first or second table is the match log
df = tables[8]

def flat(col):
    a, b = col
    a = "" if "Unnamed" in str(a) else str(a).strip()
    b = "" if "Unnamed" in str(b) else str(b).strip()
    name = "_".join([x for x in [a, b] if x]).replace(" ", "_")
    return name

def flatten(col):
    a, b = col
    a = "" if "Unnamed" in str(a) else str(a).strip()
    b = "" if "Unnamed" in str(b) else str(b).strip()
    name = "_".join([x for x in [a, b] if x]).replace(" ", "_")
    return name

df.columns = [flatten(c) for c in df.columns]

# 3. Drop totals row and clean up
df = df[df["Player"].astype(str).str.contains("Players") == False]
df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
df = df[df["Min"] >= 30]
# print(df.head())

df.to_csv("20241103_matche.csv", index=False)
print("Saved matches.csv")
