import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# load dataset
df = pd.read_csv("feature_classification/input_vectors.csv")

# prepare features (X) and labels (y)
exclude = ["Formation", "FormationLabel", "TacticalStyle", "TacticalStyleLabel", "Strategy", "StrategyLabel", "opponent", "Squad"]
X = df.select_dtypes(include=["number"]).drop(columns=[c for c in exclude if c in df.columns], errors="ignore").fillna(0)

y_form = df["FormationLabel"].astype(int)
y_tstyle = df["TacticalStyleLabel"].astype(int)
y_strategy = df["StrategyLabel"].astype(int)

n_form = y_form.nunique()
n_tstyle = y_tstyle.nunique()
n_strategy = y_strategy.nunique()

# split data into train/test
X_train, X_test, y_form_train, y_form_test, y_tstyle_train, y_tstyle_test, y_strategy_train, y_strategy_test = train_test_split(X, y_form, y_tstyle, y_strategy, test_size=0.2, random_state=42)

# normalise inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_form_train = torch.tensor(y_form_train.values, dtype=torch.long)
y_tstyle_train = torch.tensor(y_tstyle_train.values, dtype=torch.long)
y_strategy_train = torch.tensor(y_strategy_train.values, dtype=torch.long)
y_form_test = torch.tensor(y_form_test.values, dtype=torch.long)
y_tstyle_test = torch.tensor(y_tstyle_test.values, dtype=torch.long)
y_strategy_test = torch.tensor(y_strategy_test.values, dtype=torch.long)

# define model
class MultiOutputDNN(nn.Module):
    def __init__(self, input_dim, n_form, n_tstyle, n_strategy):
        super(MultiOutputDNN, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # hidden layer for each output head for specialization
        self.form_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_form))
        self.tstyle_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_tstyle))
        self.strategy_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_strategy))

    def forward(self, x):
        shared_out = self.shared(x)
        return self.form_head(shared_out), self.tstyle_head(shared_out), self.strategy_head(shared_out)

model = MultiOutputDNN(X_train.shape[1], n_form, n_tstyle, n_strategy)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
batch_size = 16

# train model
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    total_loss = 0

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_yf, batch_yt, batch_ys = X_train[indices], y_form_train[indices], y_tstyle_train[indices], y_strategy_train[indices]

        optimizer.zero_grad()
        out_f, out_t, out_s = model(batch_x)

        loss = (criterion(out_f, batch_yf) + criterion(out_t, batch_yt) + criterion(out_s, batch_ys)) / 3.0  # equally weight each task

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

# evaluate model
model.eval()
with torch.no_grad():
    out_f, out_t, out_s = model(X_test)
    pred_f = torch.argmax(out_f, dim=1)
    pred_t = torch.argmax(out_t, dim=1)
    pred_s = torch.argmax(out_s, dim=1)

print("\n=== PyTorch Multi-Output DNN Evaluation ===")
print("\n--- Formation ---")
print(classification_report(y_form_test, pred_f))
print("\n--- Tactical Style ---")
print(classification_report(y_tstyle_test, pred_t))
print("\n--- Strategy ---")
print(classification_report(y_strategy_test, pred_s, zero_division=0))



# predict strategies for all teams in match_results.csv
match_data = pd.read_csv("feature_classification/match_data/tottenham_match_results.csv")
unique_opponents = match_data["Opponent"].dropna().str.strip().str.lower().unique()
print(f"Found {len(unique_opponents)} unique opponents to predict.")

formation_map = dict(zip(df["FormationLabel"], df["Formation"]))
tstyle_map = dict(zip(df["TacticalStyleLabel"], df["TacticalStyle"]))
strategy_map = dict(zip(df["StrategyLabel"], df["Strategy"]))
results = []

for opponent_name in unique_opponents:
    # find all data rows for that opponent in the feature dataset
    sample = df[df["opponent"].str.lower() == opponent_name]
    if sample.empty:
        results.append({
            "Opponent": opponent_name.title(),
            "Predicted_Formation": "N/A",
            "Predicted_TacticalStyle": "N/A",
            "Predicted_Strategy": "N/A"
        })
        continue

    # average across all matches with this opponent
    X_sample = sample.select_dtypes(include=["number"]).drop(
        columns=[c for c in exclude if c in sample.columns],
        errors="ignore"
    ).fillna(0).mean(axis=0).to_frame().T

    # scale and convert to tensor
    X_scaled = scaler.transform(X_sample)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # predict
    with torch.no_grad():
        out_f, out_t, out_s = model(X_tensor)
        pred_f = torch.argmax(out_f, dim=1).item()
        pred_t = torch.argmax(out_t, dim=1).item()
        pred_s = torch.argmax(out_s, dim=1).item()

    results.append({
        "Opponent": opponent_name.title(),
        "Predicted_Formation": formation_map.get(pred_f, pred_f),
        "Predicted_TacticalStyle": tstyle_map.get(pred_t, pred_t),
        "Predicted_Strategy": strategy_map.get(pred_s, pred_s)
    })

# save to result.txt
with open("result.txt", "w") as f:
    f.write("=== Tactical Predictions per Unique Opponent ===\n\n")
    for r in results:
        f.write(f"Opponent: {r['Opponent']}\n")
        f.write(f"🔹 Formation: {r['Predicted_Formation']}\n")
        f.write(f"🔹 Tactical Style: {r['Predicted_TacticalStyle']}\n")
        f.write(f"🔹 Strategy: {r['Predicted_Strategy']}\n")
        f.write("-" * 45 + "\n")

print("Saved predictions for all unique opponents to result.txt!")




# predict function to predict strategy for a given opponent
def predict_strategy(opponent_name):
    opponent_name = opponent_name.strip().lower()
    sample = df[df["opponent"].str.lower() == opponent_name]
    if sample.empty:
        print(f"No data found for opponent '{opponent_name}'.")
        return None

    X_sample = sample.select_dtypes(include=["number"]).drop(
        columns=[c for c in exclude if c in sample.columns],
        errors="ignore"
    ).fillna(0)
    X_sample = scaler.transform(X_sample)
    X_sample = torch.tensor(X_sample, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        out_f, out_t, out_s = model(X_sample)
        pred_f = torch.argmax(out_f, dim=1).item()
        pred_t = torch.argmax(out_t, dim=1).item()
        pred_s = torch.argmax(out_s, dim=1).item()

    formation_map = dict(zip(df["FormationLabel"], df["Formation"]))
    tstyle_map = dict(zip(df["TacticalStyleLabel"], df["TacticalStyle"]))
    strategy_map = dict(zip(df["StrategyLabel"], df["Strategy"]))

    print(f"\n=== Tactical Recommendation vs {opponent_name.title()} ===")
    print(f"🔹 Formation: {formation_map.get(pred_f, pred_f)}")
    print(f"🔹 Tactical Style: {tstyle_map.get(pred_t, pred_t)}")
    print(f"🔹 Strategy: {strategy_map.get(pred_s, pred_s)}")

