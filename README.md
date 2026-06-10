# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project predicts whether a credit card transaction is fraudulent or legitimate using Machine Learning techniques. The dataset undergoes preprocessing, feature engineering, label encoding, feature scaling, train-test splitting, and class balancing using SMOTE before training a Random Forest Classifier.

A Streamlit web application is used to interact with the trained model and make predictions on transaction data.

## Features

* Data Cleaning and Preprocessing
* Label Encoding for Categorical Variables
* Feature Scaling using StandardScaler
* Class Balancing using SMOTE
* Random Forest Classifier
* Interactive Streamlit Web Application
* Fraud Detection Prediction

## Input Features

The model uses the following features:

* merchant
* category
* amt
* gender
* city
* state
* zip
* lat
* long
* city_pop
* job
* unix_time
* merch_lat
* merch_long
* year
* month
* day
* hour
* age
* night_transaction

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Imbalanced-Learn
* Joblib
* Streamlit

## Project Structure

CreditCardFraudDetection/

├── app.py

├── model.pkl

├── scaler.pkl

├── encoders.pkl

├── requirements.txt

├── README.md

└── Credit Card Fraud Detection ML Project.ipynb

## Installation

Clone the repository:

git clone <repository-url>

cd CreditCardFraudDetection

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## Model Workflow

1. Load Dataset
2. Perform Data Preprocessing
3. Encode Categorical Features
4. Scale Numerical Features
5. Apply SMOTE for Class Balancing
6. Train Random Forest Model
7. Save Model and Preprocessing Objects
8. Deploy Using Streamlit

## Deployment

The application is deployed using Streamlit Community Cloud.

## Author

Developed as a Machine Learning Project for Credit Card Fraud Detection.
