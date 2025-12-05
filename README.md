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
# Building Input Vectors

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