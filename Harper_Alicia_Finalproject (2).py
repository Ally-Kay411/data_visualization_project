#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# In[7]:


bike_rentals = pd.read_csv('day.csv')


# In[8]:


#check the first few rows
bike_rentals.head()


# In[9]:


# Convert 'dteday' column to datetime format
bike_rentals['dteday'] = pd.to_datetime(bike_rentals['dteday'])


# In[10]:


# Check the data types of columns
bike_rentals.info()


# In[11]:


#Exploratory Data Analysis - gain a better understanting of the structure and relationship between features/variables
bike_rentals.describe()


# In[ ]:





# In[12]:


# Plot the distribution of bike rentals
plt.figure(figsize=(10,6))
sns.histplot(bike_rentals['cnt'], bins=30, kde=True)
plt.title('Distribution of Bike Rentals')
plt.xlabel('Number of Rentals')
plt.ylabel('Frequency')
plt.show()


# In[13]:


# Correlation heatmap to look for correlations/relationships 
plt.figure(figsize=(12,8))
sns.heatmap(bike_rentals.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[14]:


# Extracting day, month, year, and weekday from 'dteday'
bike_rentals['year'] = bike_rentals['dteday'].dt.year
bike_rentals['month'] = bike_rentals['dteday'].dt.month
bike_rentals['day'] = bike_rentals['dteday'].dt.day
bike_rentals['weekday'] = bike_rentals['dteday'].dt.weekday


# In[15]:


# Drop 'dteday' column after extracting the relevant features
bike_rentals.drop('dteday', axis=1, inplace=True)


# In[16]:


# Create lag features (previous days)
bike_rentals['lag_7'] = bike_rentals['cnt'].shift(7)  # Lag of 7 days
bike_rentals['lag_30'] = bike_rentals['cnt'].shift(30)  # Lag of 30 days


# In[17]:


bike_rentals.fillna(bike_rentals.mean(), inplace=True)


# In[18]:


#rolling stats - what trends do we see over time? 
# Create rolling means over 7 and 30 days
bike_rentals['rolling_mean_7'] = bike_rentals['cnt'].rolling(window=7).mean()
bike_rentals['rolling_mean_30'] = bike_rentals['cnt'].rolling(window=30).mean()


# In[19]:


#Train and test! (60/40 split) first define the feature(x) and target/variable(y)
# Define the feature matrix X and the target variable y
X = bike_rentals.drop('cnt', axis=1)
y = bike_rentals['cnt']


# In[24]:


# Split the data into training (60%) and testing (40%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# In[25]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[27]:


# Make predictions on the test set
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

# Remove rows with missing values
bike_rentals_clean = bike_rentals.dropna()

# Recreate X and y with clean data
X = bike_rentals_clean.drop('cnt', axis=1)
y = bike_rentals_clean['cnt']

# Split again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[28]:


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# In[29]:


# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')


# In[31]:


# Create a DataFrame to display the feature names and their corresponding coefficients
coef_df = pd.DataFrame({'Coefficient': model.coef_}, index=X.columns)
print("\nModel Coefficients:")
print(coef_df)


# In[32]:


# Calculate the residuals (difference between actual and predicted values)
residuals = y_test - y_pred


# In[33]:


# Plot the residuals
plt.figure(figsize=(10,6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[34]:


# Plot residuals vs predicted values
plt.figure(figsize=(10,6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()


# In[ ]:





# In[35]:


from sklearn.linear_model import LinearRegression


# In[36]:


#Future predictions 
# Get the last row from the dataset
last_row = bike_rentals.iloc[-1:]
print (bike_rentals.iloc)


# In[37]:


# Step 2: Prepare features and target
X = bike_rentals.drop('cnt', axis=1)  # All columns except 'count'
y = bike_rentals['cnt']  # Target variable (bike rentals)


# In[42]:


# Convert categorical columns to numeric using pandas get_dummies
X_numeric = pd.get_dummies(X, drop_first=True)
y = bike_rentals['cnt']

# Handle missing values - fill with median or drop
X_numeric = X_numeric.fillna(X_numeric.median())  # Fill NaN with median
# OR
# X_numeric = X_numeric.dropna()  # Drop rows with NaN

# Split the data with numeric features
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.4, random_state=1)

# Now train the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[43]:


# Model coefficients
coef_df = pd.DataFrame({'Coefficient': model.coef_}, index=X_numeric.columns)
print("Model Coefficients:")
print(coef_df.sort_values('Coefficient', key=abs, ascending=False))

print(f"\nIntercept: {model.intercept_:.2f}")

# Performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:




