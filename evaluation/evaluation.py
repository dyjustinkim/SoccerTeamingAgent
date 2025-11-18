import json
import matplotlib.pyplot as plt
import numpy as np

def hit_at_k(predictions, real_results, k):
    hits, formhits, strathits, allhits = 0, 0, 0, 0
    total = 0
    
    for team, pred_strategy in predictions.items():
        total += 1
        top_k_strategies = list(real_results[team].keys())[:k]
        if pred_strategy in top_k_strategies:
            hits += 1
        formation, strat = pred_strategy.split()[0], pred_strategy.split()[2]
        if any(s.split()[0] == formation for s in top_k_strategies):
            formhits += 1
        if any(s.split()[2] == strat for s in top_k_strategies):
            strathits +=1 
        if any(s.split()[0] == formation for s in top_k_strategies) or any(s.split()[2] == strat for s in top_k_strategies):
            allhits += 1

    
    hit_rate = [hits, formhits, strathits, allhits]
    hit_rate = [x / total for x in hit_rate]
    return hit_rate

def compute_mrr(predictions, real_results):
    rr_scores = []

    for team, pred_strat in predictions.items():
        historical_strats = list(real_results[team].keys())

        rank, formrank, stratrank, allrank = None, None, None, None

        for i, whole_strat in enumerate(historical_strats):
            formation, strat = pred_strat.split()[0], pred_strat.split()[2]

            if whole_strat == pred_strat:
                rank = i + 1
                break
            if whole_strat.split()[0] == formation:
                if formrank == None:
                    formrank = i + 1
            if whole_strat.split()[2] == strat:
                if stratrank == None:
                    stratrank = i + 1
            if whole_strat.split()[0] == formation or whole_strat.split()[2] == strat:
                if allrank == None:
                    allrank = i + 1

        team_metrics = []
        for metric in [rank, formrank, stratrank, allrank]:
            if metric == None:
                team_metrics.append(0)
            else:
                team_metrics.append(1/metric)
        rr_scores.append(team_metrics)

    result = [sum(x) for x in zip(*rr_scores)]
    return [x / len(rr_scores) for x in result]

def average_points(predictions, real_results):
        avg_points = []

        for team, pred_strat in predictions.items():
            historical_strats = list(real_results[team].keys())

            basepoints, formpoints, stratpoints, allpoints = None, None, None, None

            for i, whole_strat in enumerate(historical_strats):
                points = real_results[team][whole_strat][1]
                formation, strat = pred_strat.split()[0], pred_strat.split()[2]

                if whole_strat == pred_strat:
                    basepoints = points
                if whole_strat.split()[0] == formation:
                    if formpoints == None:
                        formpoints = points
                if whole_strat.split()[2] == strat:
                    if stratpoints == None:
                        stratpoints = points
                if whole_strat.split()[0] == formation or whole_strat.split()[2] == strat:
                    if allpoints == None:
                        allpoints = points

            team_metrics = []
            for metric in [basepoints, formpoints, stratpoints, allpoints]:
                if metric == None:
                    team_metrics.append(0)
                else:
                    team_metrics.append(metric)
            avg_points.append(team_metrics)

        result = [sum(x) for x in zip(*avg_points)]
        return [x / len(avg_points) for x in result]

def graph_topk(k1, k2, k3):
    topk_scores = np.array([
        k1, k2, k3
    ])

    ks = [1, 3, 5] 
    methods = ["Exact", "Formation", "Strategy", "Any"]

    n_k = len(ks)  
    n_methods = topk_scores.shape[1] 
    bar_width = 0.8       
    gap = 0.15               

    x = []
    current_x = 0
    for i in range(n_k):
        cluster_positions = [current_x + j*bar_width for j in range(n_methods)]
        x.append(cluster_positions)
        current_x += n_methods*bar_width + gap

    x = np.array(x)

    plt.figure(figsize=(8,5))

    for i in range(n_methods):
        plt.bar(x[:,i], topk_scores[:,i], width=bar_width, label=methods[i])

    cluster_centers = [np.mean(pos) for pos in x]
    plt.xticks(cluster_centers, ks)
    plt.xlabel("Top-k")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.title("Top-k Accuracy Across Evaluation Methods")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def graph_single(scores, max, title):
    labels = ["Exact", "Formation", "Strategy", "Any"]

    plt.figure(figsize=(6,4))
    plt.bar(labels, scores, color='skyblue')
    plt.ylim(0, max)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

with open("evaluation/predictions.json", "r") as f:
    predictions = json.load(f)
with open("evaluation/historical_strategies.json", "r") as f:
    real_results = json.load(f)

k1 = hit_at_k(predictions, real_results, k=1)
k3 = hit_at_k(predictions, real_results, k=3)
k5 = hit_at_k(predictions, real_results, k=5)
mrr = compute_mrr(predictions, real_results)
points = average_points(predictions, real_results)

num_teams = len(real_results.keys())

avg_games, avg_strats = 0, 0
for team, whole_strats in real_results.items():
    games = sum(v[0] for v in whole_strats.values())
    unique_strats = len(whole_strats)
    avg_games += games
    avg_strats += unique_strats

#print(avg_games/num_teams, avg_strats/num_teams)

#graph_topk(k1, k3, k5)
graph_single(mrr, 1, "MRR Score")
#graph_single(points, 3, "Average Historical Points")

        



