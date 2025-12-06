import torch
import torch.nn as nn
import pandas as pd
import joblib

def predict(input_file, scaler_file, model_file, output_file):
    # load scaler
    scaler = joblib.load(scaler_file)

    # load label mappings
    df = pd.read_csv(input_file)
    formation_map = dict(zip(df["FormationLabel"], df["Formation"]))
    tstyle_map = dict(zip(df["TacticalStyleLabel"], df["TacticalStyle"]))
    strategy_map = dict(zip(df["StrategyLabel"], df["Strategy"]))
    exclude = ["Formation", "FormationLabel", "TacticalStyle", "TacticalStyleLabel",
            "Strategy", "StrategyLabel", "opponent", "Squad"]


    # define model (same as training)
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

            self.form_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_form))
            self.tstyle_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_tstyle))
            self.strategy_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_strategy))

        def forward(self, x):
            shared_out = self.shared(x)
            return (
                self.form_head(shared_out),
                self.tstyle_head(shared_out),
                self.strategy_head(shared_out)
            )


    # get input dimensions from existing dataset
    X = df.select_dtypes(include=["number"]).drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    input_dim = X.shape[1]
    n_form = df["FormationLabel"].nunique()
    n_tstyle = df["TacticalStyleLabel"].nunique()
    n_strategy = df["StrategyLabel"].nunique()

    # load trained model
    model = MultiOutputDNN(input_dim, n_form, n_tstyle, n_strategy)
    model.load_state_dict(torch.load(model_file))
    model.eval()


    # predict strategy for a given opponent
    def predict_opponent(opponent_name: str):
        sample = df[df["opponent"].str.lower() == opponent_name.lower()]

        if sample.empty:
            return {
                "Opponent": opponent_name.title(),
                "Formation": "N/A",
                "TacticalStyle": "N/A",
                "Strategy": "N/A"
            }

        # use only numeric columns and take average
        X_sample = sample.select_dtypes(include=["number"]).drop(
            columns=[c for c in exclude if c in sample.columns], errors="ignore"
        ).fillna(0).mean(axis=0).to_frame().T

        X_scaled = scaler.transform(X_sample)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            out_f, out_t, out_s = model(X_tensor)
            pred_f = torch.argmax(out_f, dim=1).item()
            pred_t = torch.argmax(out_t, dim=1).item()
            pred_s = torch.argmax(out_s, dim=1).item()

        return {
            "Opponent": opponent_name.title(),
            "Formation": formation_map.get(pred_f, pred_f),
            "TacticalStyle": tstyle_map.get(pred_t, pred_t),
            "Strategy": strategy_map.get(pred_s, pred_s)
        }


    default_teams = [
        "LeicesterCity", "Everton", "NewcastleUtd", "Arsenal", "Brentford",
        "ManchesterUtd", "Brighton", "WestHam", "CrystalPalace", "AstonVilla"
    ]


    opponents = default_teams
    print("\nUsing default opponents list.")

    print("\n=== Predictions ===\n")

    results = []

    for opp in opponents:
        pred = predict_opponent(opp)
        results.append(pred)

        print(f"Opponent: {pred['Opponent']}")
        print(f"  • Formation: {pred['Formation']}")
        print(f"  • Tactical Style: {pred['TacticalStyle']}")
        print(f"  • Strategy: {pred['Strategy']}")
        print("-" * 40)

    # save results to results.txt
    with open(output_file, "w") as f:
        f.write("=== Tactical Predictions ===\n\n")
        for r in results:
            f.write(f"Opponent: {r['Opponent']}\n")
            f.write(f"Formation: {r['Formation']}\n")
            f.write(f"Tactical Style: {r['TacticalStyle']}\n")
            f.write(f"Strategy: {r['Strategy']}\n")
            f.write("-" * 40 + "\n")

    print("\nSaved predictions to results.txt!")