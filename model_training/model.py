import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from datetime import datetime

# load dataset
df = pd.read_csv("../feature_classification/input_vectors.csv")

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
# start time
start_time = datetime.now()
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

# end time
end_time = datetime.now()
print(f'formatted start time is: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'formatted end time is: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'elapsed time is {end_time - start_time} seconds')

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


# Save scaler and pytorch model
joblib.dump(scaler, "input_scaler.pkl")
torch.save(model.state_dict(), "multi_output_dnn.pth")
print("Model and scaler saved")
