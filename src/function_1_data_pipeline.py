from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import copy
import joblib
import yaml
import joblib
from datetime import datetime

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

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def check_data(input_data, params):
    # Check data types
    assert input_data.select_dtypes("float64").columns.to_list() == params["float64_columns"], "an error occurs in float column(s)."
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."
    #assert input_data.select_dtypes("int64").columns.to_list() == params["int64_columns"], "an error occurs in int64 column(s)."
    # Check range of data
    assert set(input_data.Geography).issubset(set(params["range_Geography"])), "an error occurs in Geography range."
    assert set(input_data.Gender).issubset(set(params["range_Gender"])), "an error occurs in Gender range."
    assert input_data.CreditScore.between(params["range_CreditScore"][0], params["range_CreditScore"][1]).sum() == len(input_data), "an error occurs in CreditScore range."
    assert input_data.Age.between(params["range_Age"][0], params["range_Age"][1]).sum() == len(input_data), "an error occurs in Age range."
    assert input_data.Tenure.between(params["range_Tenure"][0], params["range_Tenure"][1]).sum() == len(input_data), "an error occurs in Tenure range."
    assert input_data.Balance.between(params["range_Balance"][0], params["range_Balance"][1]).sum() == len(input_data), "an error occurs in co range."
    assert input_data.NumOfProducts.between(params["range_NumOfProducts"][0], params["range_NumOfProducts"][1]).sum() == len(input_data), "an error occurs in NumOfProducts range."
    assert input_data.HasCrCard.between(params["range_HasCrCard"][0], params["range_HasCrCard"][1]).sum() == len(input_data), "an error occurs in HasCrCard range."
    assert input_data.IsActiveMember.between(params["range_IsActiveMember"][0], params["range_IsActiveMember"][1]).sum() == len(input_data), "an error occurs in IsActiveMember range."
    assert input_data.EstimatedSalary.between(params["range_EstimatedSalary"][0], params["range_EstimatedSalary"][1]).sum() == len(input_data), "an error occurs in EstimatedSalary range."

if __name__ == "__main__":
    # 1. Load configuration file
    #config_data = util.load_config()
    config_data = load_config()
    
    #2. Read all raw Dataset
    raw_dataset = read_raw_data(config_data).drop(["RowNumber","CustomerId","Surname"], axis = 1)
    
    #3.Check Dataset
    check_data(raw_dataset, config_data)
    
    # 3. Reset index
    raw_dataset.reset_index(
        inplace = True,
        drop = True
    )
    
    # 13. Splitting input output
    # Pemisahan Variabel X dan Y
    X = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset.Exited.copy()

    # 14. Splitting train test
    #Split Data 80% training 20% testing
    X_train, X_test, \
    y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.3, 
        random_state = 123)
    
    # 15. Splitting test valid
    X_valid, X_test, \
    y_valid, y_test = train_test_split(
        X_test, y_test,
        test_size = 0.5,
        random_state = 42,
        stratify = y_test
    )
    
    #Menggabungkan x train dan y train untuk keperluan EDA
    pickle_dump(X_train, config_data["train_set_path"][0])
    pickle_dump(y_train, config_data["train_set_path"][1])

    pickle_dump(X_valid, config_data["valid_set_path"][0])
    pickle_dump(y_valid, config_data["valid_set_path"][1])

    pickle_dump(X_test, config_data["test_set_path"][0])
    pickle_dump(y_test, config_data["test_set_path"][1])