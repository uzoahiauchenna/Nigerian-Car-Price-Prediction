## Project Description

This project is a Nigerian Car Price Prediction System built as a capstone 
project for an Artificial Intelligence and Machine Learning programme. 
The goal is to predict the market price of a car in Nigeria based on its 
specifications and condition using supervised machine learning.

### Problem Statement
The Nigerian used car market lacks a reliable and transparent pricing system. 
Buyers and sellers often rely on guesswork or informal negotiations which 
leads to unfair pricing. This project addresses that problem by building a 
data-driven price prediction model that estimates fair market value based on 
historical listing data.

### Dataset
The dataset contains 3,722 Nigerian car listings scraped from online 
marketplaces with features including make, model, year of manufacture, 
mileage, engine size, fuel type, gear type, condition, colour and car body 
type. Extensive data cleaning was performed including handling missing values, 
correcting inconsistent entries, engineering new features and removing outliers.

### Methodology
- **Data Cleaning** — handled missing values, corrected car make and model 
  entries, mapped car body types and removed corrupted listings
- **Exploratory Data Analysis** — analysed price distributions, feature 
  correlations and market trends using histograms, box plots, scatter plots 
  and heatmaps
- **Feature Engineering** — encoded categorical variables using Label 
  Encoding and applied log transformation to the target variable to handle 
  price skewness
- **Model Training** — trained and compared four models: Gradient Boosting, 
  XGBoost, Random Forest and LightGBM
- **Hyperparameter Tuning** — optimised the best model using GridSearchCV 
  with 5-fold cross validation
- **Deployment** — built an interactive web application using Streamlit

### Results
The Gradient Boosting Regressor achieved the best performance:
- Test R²  : 71.8%
- Test MAE : ₦1,032,191
- Test RMSE: ₦2,185,083

The model performs strongest in the ₦1M – ₦15M price range which represents 
the majority of the Nigerian car market. Log transformation of the target 
variable significantly improved predictions for high-end vehicles above ₦15M.

### Key Findings
- Year of manufacture is the single strongest predictor of car price at 42.7% 
  feature importance — older cars are significantly cheaper
- Engine size is the second most important feature at 28.3%
- Nigerian Used cars are on average 40% cheaper than Foreign Used equivalents
- Toyota dominates the Nigerian market with 43% of all listings
- SUVs command the highest median prices across all car body types

### Web Application
The Streamlit application allows users to input car specifications and receive 
an instant price estimate with a ±15% confidence range. The interface includes 
a model performance dashboard showing evaluation metrics and feature importance 
analysis.

### Limitations
- Dataset limited to 3,722 listings which affects prediction accuracy for 
  rare or luxury vehicles
- Prices reflect historical listing data and may not account for current 
  exchange rate fluctuations affecting the Nigerian car market
- CNN image classifier for automatic car make detection from photos is still 
  in development

### Technologies Used
- Python, Pandas, NumPy
- XGBoost, LightGBM
- Matplotlib, Seaborn
- Streamlit
- Joblib
