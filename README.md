# Kaggle_New York City Taxi Trip Duration
Machine Learning Solution for [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)



The contest is using the given data, which includes pickup time, geo-coordinates, number of passengers, and several other variables, 
to build a model that predicts the total ride duration of taxi trips in New York City.  

This repository demonstrates some feature extraction ideas and machine learning practices. 

The best solution I found contained SVR, Gradient Boost Regressor, and XGBoost models, 
with AdaBoost as ensemble model. The Public LB score is **0.32 (23rd of 1257, top 2 %)**.

# Instruction
*	download data from the [original dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) and [additional resource](https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm)
*	the ipython notebook `./EDA.ipynb` show the basic exploration of the original features.  
* run python `./feature_enginerring.py` to get the features 
*	run python `./model.py` to generate the best predicting result.
