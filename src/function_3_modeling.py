#import all realated libraries
#import libraries for data analysis
import numpy as np
import pandas as pd

# import library for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# import pickle and json file for columns and model file
import pickle
import json
import joblib
import yaml
import scipy.stats as scs

# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")

# library for model selection and models
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

# evaluation metrics for classification model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import GridSearchCV
import json
from datetime import datetime
from sklearn.metrics import classification_report
import uuid

from tqdm import tqdm
import pandas as pd
import os
import copy
import yaml
import joblib

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

def load_valid_clean(params: dict) -> pd.DataFrame:
    # Load valid set
    X_valid_clean = pickle_load(params["standar_scaler_valid"][0])
    y_valid = pickle_load(params["standar_scaler_valid"][1])

    return X_valid_clean, y_valid

def load_test_clean(params: dict) -> pd.DataFrame:
    # Load test set
    X_test_clean = pickle_load(params["standar_scaler_test"][0])
    y_test = pickle_load(params["standar_scaler_test"][1])

    return X_test_clean, y_test

def load_smote_clean(params: dict) -> pd.DataFrame:
    # Load test set
    X_sm_clean = pickle_load(params["standar_scaler_sm"][0])
    y_sm = pickle_load(params["standar_scaler_sm"][1])

    return X_sm_clean, y_sm


def load_dataset(params: dict) -> pd.DataFrame:
    # Debug message
    print_debug("Loading dataset.")
    
    # load sm set
    X_sm_clean, y_sm =  load_smote_clean(params)

    # Load valid set
    X_valid_clean, y_valid = load_valid_clean(params)

    # Load test set
    X_test_clean, y_test = load_test_clean(params)
    
    # Debug message
    print_debug("Dataset loaded.")

    # Return the dataset
    return  X_sm_clean, y_sm, X_valid_clean, y_valid, X_test_clean, y_test

def load_data_scaling(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_sm_clean = pickle_load(config_data["standar_scaler_sm"][0])
    y_sm = pickle_load(config_data["standar_scaler_sm"][1])

    X_test_clean = pickle_load(config_data["standar_scaler_test"][0])
    y_test = pickle_load(config_data["standar_scaler_test"][1])

    X_valid_clean = pickle_load(config_data["standar_scaler_valid"][0])
    y_valid = pickle_load(config_data["standar_scaler_valid"][1])

    # Return 3 set of data
    return X_sm_clean, y_sm, X_valid_clean, y_valid, X_test_clean, y_test

def binary_classification_xgb_tuned(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # set hyperparameters for tuning
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.5, 1.0],
    'subsample': [0.5, 0.7, 1.0]
    }

    # instantiate the classifier
    xgb_clf = xgb.XGBClassifier(random_state=123)
    
    # perform grid search to find best hyperparameters
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(x_train, y_train)
    print('Best hyperparameters:', grid_search.best_params_)
    
    # create classifier with best hyperparameters
    best_xgb_clf = xgb.XGBClassifier(**grid_search.best_params_, random_state=123)
    
    # train the model
    best_xgb_clf.fit(x_train, y_train)
    
    # evaluate on validation set
    valid_pred = best_xgb_clf.predict(x_valid)
    valid_acc = accuracy_score(y_valid, valid_pred)
    print('Validation accuracy:', valid_acc)
    
    # evaluate on test set
    test_pred = best_xgb_clf.predict(x_test)
    test_acc = accuracy_score(y_test, test_pred)
    print('Test accuracy:', test_acc)
    
    return best_xgb_clf

def save_model_log(model, model_name, X_test, y_test):
    # generate unique id
    model_uid = uuid.uuid4().hex
    
    # get current time and date
    now = time_stamp()
    training_time = now.strftime("%H:%M:%S")
    training_date = now.strftime("%Y-%m-%d")
    
    # generate classification report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # create dictionary for log
    log = {"model_name": model_name,
           "model_uid": model_uid,
           "training_time": training_time,
           "training_date": training_date,
           "classification_report": report}
    
    # menyimpan log sebagai file JSON
    with open('training_log/training_log.json', 'w') as f:
        json.dump(log, f)
        
if __name__ == "__main__":
    # 1. Load configuration file
    config_data = load_config()
    
    # 2. Load dataset
    X_sm_clean, y_sm, X_valid_clean, y_valid, X_test_clean, y_test = load_data_scaling(config_data)
    
    xgb_best = binary_classification_xgb_tuned(x_train = X_sm_clean, y_train = y_sm, \
                                               x_valid = X_valid_clean, y_valid = y_valid, \
                                               x_test = X_test_clean, y_test = y_test)
    
    save_model_log(model = xgb_best, model_name = "XGBoost CV", X_test = X_test_clean, y_test=y_test)
    
    xgboost_cv = config_data["model_final"]
    with open(xgboost_cv, 'wb') as file:
        pickle.dump(xgb_best, file)