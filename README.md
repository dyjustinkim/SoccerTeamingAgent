# SoccerTeamingAgent
Program to Predict Soccer Formations with Machine Learning

# Data Collection
# Feature Classification
Collected data can be classified into features by running ```feature_classification/boruta_classification.py``` script with the saved player data csv file.
```
python feature_classification/boruta_classification.py
```
Once run, the classified features will be saved in ```boruta_confirmed_features.csv``` file.
# Feature Segmentation
Player features are segmented into sub-roles using K-means clustering by running the segmentation script:
```
python feature_segmentation/kmeans_clustering.py
```
Once run, each player is assigned a segmented position (e.g., CB0, CMF1, FW0), and the results are saved to:
```
player_clusters_by_position.csv
```
# Building Input Vectors

Match level player statistics can be collected by downloading the html file from fbref.com --> Searching for a team --> Scores and Fixtures --> Match Report

Run the three files under new_team_input_vector_creation folder in order

```
python new_team_input_vector_creation/match_level_HTML_parser.py
python new_team_input_vector_creation/match_level_processer.py
python new_team_input_vector_creation/match_level_input_vector_creation.py 
```

Once run, each player ia assigned a segmented posiion for that match, and the results will be saved to: 
```
player_match_with_segments.csv
```

# Model Training
Train the model using the input vectors by running ```model.py``` script
```
python model.py
```
Once run, evaluation results will be printed and the model and scaler will be saved for future use
model path: multi_output_dnn.pth
scaler path: input_scaler.pkl

# Prediction
Run the ```predict.py``` script and input the teams (comma-separated) that you want prediction for
```
python predict.py
```
