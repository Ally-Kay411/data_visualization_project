# Summary of Comprehensive Machine Learning Project: Predicting Bike Rentals Over a 30-Day Period
_________________________________________
In this project, the goal is to forecast the number of bike rentals over a 30-day period, using historical data from the Bike Sharing dataset (day.csv). The problem is framed as a time series forecasting task, where multiple linear regression is employed to predict bike rentals based on various features.
The project focuses on:
1. Data Preprocessing: Preparing and cleaning the dataset, handling missing values, and encoding categorical variables.
2. Feature Engineering: Selecting relevant features and transforming data to improve model performance.
3. Time Series Handling: Using a 30-day rolling window approach to account for temporal dependencies and improve prediction accuracy.
4. Model Evaluation: Assessing model performance with key metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) to measure the model's predictive accuracy and fit.
5. Forecasting: Predicting bike rentals over a 30-day future period and generating forecasts based on the trained model.
By incorporating a rolling window for predictions, the project ensures that predictions are continuously updated based on the most recent data, allowing for more accurate forecasting in the dynamic environment of bike rentals.

__________________________________________
I was able to accomplish the following:

* Good feature engineering: Adding lag features, rolling means, and time-based features
* Proper Data Preprocessing: Handled missing values, and conversted categorical variables
* Model Building: Successfully trained a linear regression model
* Time Series Elements

__________________________________________

However I'm still working on my model results ⚠️
The model is showing perfect performance (R² = 1.0, MAE = 0.00, RMSE = 0.00), which indicates data leakage
