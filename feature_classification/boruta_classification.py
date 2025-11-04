import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# just run this file, replacing DATASET_PATH with whatever dataset is necessary
# the important features are in confirmed

DATASET_PATH = "sampledata/tottenham2425.csv"

# load the dataset using pandas
df = pd.read_csv(DATASET_PATH)

# get the position and name from the player column
def get_position_and_name(player_data):
    if type(player_data) is not str:
        return 'N/A', 'N/A'
    player_data = player_data.split(' ')

    position, name = '', ''

    # extract position and name from player data
    position = player_data[-1] if len(player_data) > 1 else 'N/A'
    name_arr = player_data[:-1] if len(player_data) > 1 else player_data
    name = ' '.join(n for n in name_arr if not n.isnumeric())

    # extract position: these positions are based on Lee paper https://peerj.com/articles/cs-853/

    first_position = position.split(',')[0]
    main_position, sub_position = '', ''
    # for the purpose of our assignment, we will only focus on the first position
    if '(' in first_position and ')' in first_position:
        main_position, sub_position = first_position.split('(')
        sub_position = sub_position[:-1]
    else:
        main_position, sub_position = first_position, ''

    if main_position == 'GK':
        position = 'GK'
    elif main_position == 'D' and ('R' in sub_position or 'L' in sub_position):
        position = 'WB'
    elif main_position == 'D' and 'C' in sub_position:
        position = 'CB'
    elif 'DM' in main_position:
        position = 'DMF'
    elif main_position == 'M':
        position = 'CMF'
    elif main_position == 'AM' and len(sub_position) == 1 and sub_position == 'C':
        position = 'AMF'
    elif main_position == 'AM' and ('R' in sub_position or 'L' in sub_position):
        position = 'Wing'
    elif main_position == 'FW':
        position = 'FW'

    return position, name

# assign them as the new position and name columns
results = [get_position_and_name(player) for player in df['Player.1']]
df['Position'] = [r[0] for r in results]
df['Name'] = [r[1] for r in results]

# We will make sure to process only the numeric columns for our Boruta training
# First, replace every '-' value with np.nan, since numpy cannot process '-' values
pd.set_option('future.no_silent_downcasting', True)
numeric_df = df.replace('-', np.nan)
string_columns = ['Player', 'Player.1', 'Name', 'Position']

# convert every number to pandas numeric numbers, coercing empty values to NaN
for column in numeric_df.columns:
    if column not in string_columns:
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors='coerce')


# choose target classification
df['Target'] = df['Position']

# Prepare feature matrix
feature_cols = [c for c in df.columns if c not in ['Player', 'Player.1', 'Name', 'Position', 'Target']]
X = numeric_df[feature_cols].fillna(0).values
y = df['Target'].values

# set up random forest and boruta classifier
rf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42, verbose=2, max_iter=200)
boruta_selector.fit(X, y)

# Get feature importance results based on Boruta flags
confirmed = np.array(feature_cols)[boruta_selector.support_]
weak = np.array(feature_cols)[boruta_selector.support_weak_]
rejected = []
for feature, is_confirmed, is_weak in zip(feature_cols, boruta_selector.support_, boruta_selector.support_weak_):
    if not is_confirmed and not is_weak:
        rejected.append(feature)

rejected = np.array(rejected)

print(f"Confirmed features: {confirmed}")
print(f"Weak features: {weak}")
print(f"Confirmed and weak features: {confirmed + weak}")
print(f"Rejected features: {rejected}")

# Print out distribution of the positions relative to the confirmed and weak features
# (What is the score of the features for each position)
df_confirmed_and_weak = df[['Target'] + list(confirmed) + list(weak)].copy()

for c in confirmed:
    df_confirmed_and_weak[c] = pd.to_numeric(df_confirmed_and_weak[c], errors='coerce')

summary = df_confirmed_and_weak.groupby('Target')[confirmed].mean().T.fillna(0)
print(f'Summary: {summary}')

# Save the confirmed and weak features to a CSV for clustering step
output_path = "boruta_confirmed_features.csv"
df_confirmed_and_weak.to_csv(output_path, index=False)
print(f"Saved Boruta feature data to {output_path}")
