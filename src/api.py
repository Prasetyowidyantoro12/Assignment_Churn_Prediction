from fastapi import FastAPI, Form
from pydantic import BaseModel
import pandas as pd
from joblib import load
import joblib
#import function_1_data_pipeline as function_1_data_pipeline
#import function_2_data_processing as function_2_data_processing
#import function_3_modeling as function_3_modeling
import src.function_1_data_pipeline as function_1_data_pipeline
import src.function_2_data_processing as function_2_data_processing
import src.function_3_modeling as function_3_modeling
#import src.util as util
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from tqdm import tqdm
import os
import copy
import yaml
from datetime import datetime
import uvicorn
import sys

config_dir = "config/config.yaml"

def time_stamp() -> datetime:
    # Return current date and time
    return datetime.now()

def load_config() -> dict: 
    # Try to load yaml file
    try:
        with open(config_dir, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fe:
        raise RuntimeError("Parameters file not found in path.")

    # Return params in dict format
    return config

def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    # Dump data into file
    joblib.dump(data, file_path)

params = load_config()
PRINT_DEBUG = params["print_debug"]

def print_debug(messages: str) -> None:
    # Check wheter user wants to use print or not
    if PRINT_DEBUG == True:
        print(time_stamp(), messages)

###################################################
#API
app = FastAPI() 

config_data = load_config()
# 2. Load dataset
X_train, X_valid, X_test, y_train, y_valid, y_test = function_2_data_processing.load_data(config_data)
#3. Load dataset
dataset, valid_set, test_set = function_2_data_processing.load_dataset(config_data)
#Scaler
scaler = function_2_data_processing.load_scaler(config_data["scaler"])
# Load model and make prediction])
model = joblib.load(config_data["model_final"])
# model = joblib.load('model/5 - Model Final/xgboost_cv.pkl')

class api_data(BaseModel):
    Geography: str
    Gender: str
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    IsActiveMember: int
    HasCrCard: int
    EstimatedSalary: float

@app.get("/")
def home():
    return "Hello, FastAPI up!"    

@app.post("/predict/")
def predict(data: api_data):
    # Convert data api to dataframe
    config_data = load_config()
    #Input data
    df = pd.DataFrame(data.dict(), index=[0])
    
    #Data Defense
    try:
        function_1_data_pipeline.check_data(df, config_data)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}

    # Get Dummies for Categorical Columns
    dataset, df = function_2_data_processing.get_dummies(X_train, df)
    
    # Standart Scaler
    df = function_2_data_processing.transform_data(df, scaler)
    
    #Sort Columns
    df = df[sorted(df.columns)]
    
    # Make prediction
    prediction = model.predict(df)
    # Check prediction result
    if prediction[0] == 0:
        return "Class 0 = Customer stay"
    else:
        return "Class 1 = Customer churn"
    
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

#    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
# "uvicorn src.api:app --reload"    
#Contoh bisa digunakan
"""
{
  "Geography": "Germany",
  "Gender": "Female",
  "CreditScore": 500,
  "Age": 60,
  "Tenure": 3,
  "Balance": 34562,
  "NumOfProducts": 2,
  "IsActiveMember": 0,
  "HasCrCard": 0,
  "EstimatedSalary": 11267
}
"""