 # Nigerian Car Price Prediction

A machine learning project that predicts car prices in the Nigerian 
market using Gradient Boosting Regressor.

## Project Structure
- `Price Pred.ipynb` — main notebook with data cleaning, EDA and model training
- `app.py` — Streamlit web application
- `car_prices_cleaned.csv` — cleaned dataset
- `car_price_model.pkl and car_price_model2.pkl` — trained price prediction model
- `encoders.pkl and encoders2.pkl` — saved label encoders

## Model Performance
- Gradient Boosting Regressor — Test R²: 79.5%
- Mean Absolute Error: ₦785,950

## Features
- Input car details to get an estimated market price in Nigeria
- Model performance dashboard with feature importance analysis
- EDA visualisations

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Requirements
- Python 3.12
- See requirements.txt for full list

## Limitations
- CNN image classifier for auto make detection is still in development
- Model performs best in the ₦1M–₦10M price range
- Dataset limited to 3,722 listings
