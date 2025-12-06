import time
from initial_scraper import scrape_player_data
import os
from feature_classification.boruta_classification import boruta_classification
from feature_classification.kmeans_clustering import kmeans_clustering
from match_scraper import scrape_season_fixtures
from match_scraper import get_match_results
from match_scraper import get_league_table
from feature_classification.matchinputformat import matchinputformat
from feature_classification.matchlevelinputs import matchlevelinputs
from feature_classification.team_vector_build import team_vector_build
from model_training.model import model
from prediction.predict import predict




def run_with_retries(scraper_func, url, filename, param2, max_attempts=3):
    
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        
        try:
            data = scraper_func(url, filename, param2)
            return data
            
        except Exception as e:
            print(f"Failure detected: {type(e).__name__}: {e}")
            if attempt >= max_attempts:
                print("Max fail aattempts reach. Please restart.")

def main():
    new_folder_path = os.path.join(os.getcwd(), "tot24-25")
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Directory '{new_folder_path}' created or already exists.")
    except OSError as e:
        print(f"Error creating directory: {e}")

    if os.path.exists(os.path.join(new_folder_path, "initialteam.csv")):
        pass
    else:
        run_with_retries(scrape_player_data, "https://www.whoscored.com/teams/30/archive/england-tottenham", 
                        "Premier League - 2024/2025", os.path.join(new_folder_path, "initialteam.csv"))

    if os.path.exists(os.path.join(new_folder_path, "boruta_confirmed_features.csv")):
        pass
    else:
        boruta_classification(os.path.join(new_folder_path, "initialteam.csv"), os.path.join(new_folder_path, "boruta_confirmed_features.csv"))

    if os.path.exists(os.path.join(new_folder_path, "player_clusters_by_position.csv")):
        pass
    else:
        kmeans_clustering(os.path.join(new_folder_path, "boruta_confirmed_features.csv"), os.path.join(new_folder_path, "player_clusters_by_position.csv"))

    if os.path.exists(os.path.join(new_folder_path, "match_data")):
        pass
    else:
        run_with_retries(scrape_season_fixtures, "https://fbref.com/en/squads/361ca564/2024-2025/matchlogs/c9/schedule/Tottenham-Hotspur-Scores-and-Fixtures-Premier-League", 
                        None, os.path.join(new_folder_path, "match_data"))

    if os.path.exists(os.path.join(new_folder_path, "player_match_logs.csv")):
        pass
    else:
        matchinputformat(os.path.join(new_folder_path, "match_data"), os.path.join(new_folder_path, "player_match_logs.csv"))

    if os.path.exists(os.path.join(new_folder_path, "player_match_with_segments.csv")):
        pass
    else:
        matchlevelinputs(os.path.join(new_folder_path, "player_match_logs.csv"), os.path.join(new_folder_path, "player_clusters_by_position.csv"), os.path.join(new_folder_path, "player_match_with_segments.csv"))

    if os.path.exists(os.path.join(new_folder_path, "player_match_logs.csv")):
        pass
    else:
        matchinputformat(os.path.join(new_folder_path, "match_data"), os.path.join(new_folder_path, "player_match_logs.csv"))

    get_match_results("https://fbref.com/en/squads/361ca564/2024-2025/matchlogs/c9/schedule/Tottenham-Hotspur-Scores-and-Fixtures-Premier-League",
                      os.path.join(new_folder_path, "match_data", "match_results.csv"))
    get_league_table("https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats",
                    os.path.join(new_folder_path, "match_data", "league_table.csv"))

    if os.path.exists(os.path.join(new_folder_path, "team_match_vectors.csv")):
        pass
    else:
        team_vector_build(os.path.join(new_folder_path, "player_match_with_segments.csv"), 
                          os.path.join(new_folder_path, "match_data", "league_table.csv"),
                          os.path.join(new_folder_path, "match_data", "match_results.csv"),
                          os.path.join(new_folder_path, "team_match_vectors.csv"))
        
    if os.path.exists(os.path.join(new_folder_path, "model_training")):
        pass
    else:
        try:
    
            os.makedirs(os.path.join(os.getcwd(), new_folder_path, "model_training"), exist_ok=True)
            print(f"Directory model_training created or already exists.")
        except OSError as e:
            print(f"Error creating directory: {e}")
        model(os.path.join(new_folder_path, "team_match_vectors.csv"), 
              os.path.join(new_folder_path, "model_training", "input_scaler.pkl"),
              os.path.join(new_folder_path, "model_training", "multi_output_dnn.pth"))
        

    if os.path.exists(os.path.join(new_folder_path, "predictions")):
        pass
    else:
        try:
    
            os.makedirs(os.path.join(os.getcwd(), new_folder_path, "prediction"), exist_ok=True)
            print(f"Directory prediction created or already exists.")
        except OSError as e:
            print(f"Error creating directory: {e}")
        predict(os.path.join(new_folder_path, "team_match_vectors.csv"), 
              os.path.join(new_folder_path, "model_training", "input_scaler.pkl"),
              os.path.join(new_folder_path, "model_training", "multi_output_dnn.pth"),
              os.path.join(new_folder_path, "prediction", "results.txt"))
    
main()