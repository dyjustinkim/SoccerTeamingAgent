import pandas as pd
import json
import os

def evaluate_historical(file_name, output_name):
    df = pd.read_csv(file_name)
    all_teams_strats = {}
    for index, row in df.iterrows():
        team_name = row["team2 name"]
        if team_name not in all_teams_strats:
            all_teams_strats[team_name] = {}

        # Obtain full name of strategy (formation, tactics, and strategy)
        team_strats = all_teams_strats[team_name]
        formation = row["team1 formation"]
        tactics = row["team1 tactics"]
        strat = row["team1 strat"]
        full_strat = f"{formation} {tactics} {strat}"

        # Calculate number of strategy appearances, points, goal difference, xGD
        gd = row["team1 goals"] - row["team2 goals"]
        xgd = row["team1 xG"] - row["team2 xG"]
        points = row["team1 points"]
        new_strat = [1, points, gd, xgd]

        # Incrememnt statitics if strategy used multiple times
        if full_strat not in team_strats:
            all_teams_strats[team_name][full_strat] = new_strat
        else:
            all_teams_strats[team_name][full_strat] = [a + b for a,b in 
                                                       zip(new_strat, team_strats[full_strat])]

    # Average statistics across all matches and rank by stats
    for team in list(all_teams_strats.keys()):
        d = all_teams_strats[team]
        d_normalized = {
            k: [v[0]] + [x / v[0] for x in v[1:]]  
            for k, v in d.items()
        }
        all_teams_strats[team] = dict(sorted(d_normalized.items(), 
                                             key=lambda item: 
                                             (item[1][1], item[1][2]), reverse=True))


    directory = "evaluation"   
    filename = output_name
    filepath = os.path.join(directory, filename)

    os.makedirs(directory, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(all_teams_strats, f, indent=4)


        
    


evaluate_historical("evaluation/Prem_22-23_clustered.csv", "historical_strategies.json")