from src.function_1_data_pipeline import read_raw_data, check_data
from src.function_2_data_processing import imputeData, get_dummies, sm_fit_resample, fit_scaler, load_scaler, transform_data
from src.function_3_modeling import *
#from fungsi_4_modeling import load_smote_clean, load_valid_clean, load_test_clean, binary_classification_xgb_tuned, save_model_log
import pandas as pd
from numpy import nan
from sklearn.impute import SimpleImputer
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
#import src.util as util
from sklearn.datasets import make_classification
import os
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import uuid
from datetime import datetime
import json
from sklearn.metrics import classification_report
import pickle
import joblib
import yaml

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

config_data = load_config()

#####################################
### READ DATA DAN CHECK TIPE DATA ###
#####################################
def test_read_raw_data():
    config = {"raw_dataset_dir": "dataset/1 - raw data/"}
    df = read_raw_data(config)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_check_data():
    # define test data
    input_data = pd.DataFrame({
        'Geography': ['France', 'Germany'],
        'Gender': ['Male', "Male"],
        'CreditScore': [600, 800],
        'Age': [35, 42],
        'Tenure': [2, 5],
        'Balance': [10000.62, 20000.53],
        'NumOfProducts': [2, 1],
        'HasCrCard': [1, 0],
        'IsActiveMember': [1, 0],
        'EstimatedSalary': [60000.98, 70000.65]
        #'Exited': [1, 0]
    })
    params = {
        'float64_columns': ['Balance', 'EstimatedSalary'],
        'int64_columns': ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited'],
        'object_columns': ['Geography', 'Gender'],
        'range_Geography': ['France', 'Germany', 'Spain'],
        'range_Gender': ['Female', 'Male'],
        'range_CreditScore': [300, 850],
        'range_Age': [18, 100],
        'range_Tenure': [0, 10],
        'range_Balance': [0, 250000],
        'range_NumOfProducts': [1, 4],
        'range_HasCrCard': [0, 1],
        'range_IsActiveMember': [0, 1],
        'range_EstimatedSalary': [0, 1000000]
        #'Exited_categories': [0, 1]
    }
    
    # run the function to be tested
    check_data(input_data, params)

###################
## IMPUTASI DATA ##
###################
#IMPUTASI DATA
def test_imputeData():
    # Arrange
    data = pd.DataFrame({
        'CreditScore': [520, 451, 621, np.nan, np.nan],
        'Age': [28, 33, 31, 41, 45],
        'Tenure': [2, 5, 6, 3, 8],
        'Balance': [47522.07, 2214.0, 4521.0, 2312.43, 129264.05],
        'NumOfProducts': [2, 1, 2, 1, 2],
        'HasCrCard': [0, 0, 1, 1, 0],
        'IsActiveMember': [1, 1, 1, 0, 0],
        'EstimatedSalary': [181534.04, 135502.77, 168261.4, 44937.01, 19150.14],
        'Geography': ['France', 'Spain', 'Germany', np.nan, np.nan],
        'Gender': ['Female', 'Male', 'Male', np.nan, np.nan]
    })
    numerical_columns_mean = ['CreditScore', 'Balance', 'EstimatedSalary']
    numerical_columns_median = ['Age']
    categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'NumOfProducts', 'Tenure', 'IsActiveMember']
    
    
    X_train_impute = imputeData(data = data, 
                                numerical_columns_mean = numerical_columns_mean, 
                                numerical_columns_median = numerical_columns_median, 
                                categorical_columns = categorical_columns)

    # Assert
    assert X_train_impute.isnull().sum().sum() == 0

"""
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
                         'CreditScore': [648, 693, 586, 438, 768],
                         'Geography': ['Spain', 'Spain', 'Spain', 'Germany', 'Germany'],
                         'Gender': ['Male', np.nan, 'Female', 'Male', 'Female'],
                         'Age': [55, 57, 33, np.nan, 43],
                         'Tenure': [1, 9, 7, 8, 2],
                         'Balance': [81370.07, 0.0, 0.0, np.nan, 129264.05],
                         'NumOfProducts': [1, 2, np.nan, 1, 2],
                         'HasCrCard': [0, 1, 1, 1, 0],
                         'IsActiveMember': [1, 1, 1, 0, 0],
                         'EstimatedSalary': [181534.04, 135502.77, 168261.4, 44937.01, 19150.14]
                        })

    return data


def test_imputeData(sample_data):
    numerical_columns = ['CreditScore', 'Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
    categorical_columns = ['Geography', 'Gender']
    imputed_data = imputeData(sample_data, numerical_columns, categorical_columns)
    
    assert imputed_data.isna().sum().sum() == 0
    assert imputed_data.shape == sample_data.shape
"""
#####################################
########## GET DUMMIES DATA #########
#####################################
@pytest.fixture
def test_get_dummies():
    train_df = pd.DataFrame({
        'CreditScore': [648, 693, 586, 438, 768],
        'Age': [55, 57, 33, 24, 43],
        'Tenure': [1, 9, 7, 8, 2],
        'Balance': [81370.07, 0.0, 0.0, 2312.43, 129264.05],
        'NumOfProducts': [1, 2, 2, 1, 2],
        'HasCrCard': [0, 1, 1, 1, 0],
        'IsActiveMember': [1, 1, 1, 0, 0],
        'EstimatedSalary': [181534.04, 135502.77, 168261.4, 44937.01, 19150.14],
        'Geography': ['Spain', 'France', 'Germany', 'Spain'],
        'Gender': ['Male', 'Male', 'Female', 'Female']
    })
    input_df = pd.DataFrame({
        'CreditScore': [520, 451, 621, 524, 481],
        'Age': [28, 33, 31, 41, 45],
        'Tenure': [2, 5, 6, 3, 8],
        'Balance': [47522.07, 2214.0, 4521.0, 2312.43, 129264.05],
        'NumOfProducts': [2, 1, 2, 1, 2],
        'HasCrCard': [0, 0, 1, 1, 0],
        'IsActiveMember': [1, 1, 1, 0, 0],
        'EstimatedSalary': [181534.04, 135502.77, 168261.4, 44937.01, 19150.14],        
        'Geography': ['France', 'Spain', 'Germany'],
        'Gender': ['Female', 'Male', 'Male']
    })
    expected_train_dummies = pd.DataFrame({
        'CreditScore': [648, 693, 586, 438, 768],
        'Age': [55, 57, 33, 24, 43],
        'Tenure': [1, 9, 7, 8, 2],
        'Balance': [81370.07, 0.0, 0.0, 2312.43, 129264.05],
        'NumOfProducts': [1, 2, 2, 1, 2],
        'HasCrCard': [0, 1, 1, 1, 0],
        'IsActiveMember': [1, 1, 1, 0, 0],
        'EstimatedSalary': [181534.04, 135502.77, 168261.4, 44937.01, 19150.14],
        'Geography_France': [0, 1, 0, 0],
        'Geography_Germany': [0, 0, 1, 0],
        'Geography_Spain': [1, 0, 0, 1],
        'Gender_Female': [0, 0, 1, 1],
        'Gender_Male': [1, 1, 0, 0]
    })
    expected_input_dummies = pd.DataFrame({
        'CreditScore': [520, 451, 621, 524, 481],
        'Age': [28, 33, 31, 41, 45],
        'Tenure': [2, 5, 6, 3, 8],
        'Balance': [47522.07, 2214.0, 4521.0, 2312.43, 129264.05],
        'NumOfProducts': [2, 1, 2, 1, 2],
        'HasCrCard': [0, 0, 1, 1, 0],
        'IsActiveMember': [1, 1, 1, 0, 0],
        'EstimatedSalary': [181534.04, 135502.77, 168261.4, 44937.01, 19150.14], 
        'Geography_France': [1, 0, 0],
        'Geography_Germany': [0, 0, 1],
        'Geography_Spain': [0, 1, 0],
        'Gender_Female': [1, 0, 0],
        'Gender_Male': [0, 1, 1]
    })
    
    train_dummies, input_dummies = get_dummies(train_df, input_df)
    
    pd.testing.assert_frame_equal(train_dummies, expected_train_dummies)
    pd.testing.assert_frame_equal(input_dummies, expected_input_dummies)

    
######################
### BALANCING DATA ###
######################
@pytest.fixture
def test_sm_fit_resample():
    data = pd.DataFrame({
                        'CreditScore': [648, 693, 586, 438, 768],
                        'Age': [55, 57, 33, 24, 43],
                        'Tenure': [1, 9, 7, 8, 2],
                        'Balance': [81370.07, 0.0, 0.0, 2312.43, 129264.05],
                        'NumOfProducts': [1, 2, 2, 1, 2],
                        'HasCrCard': [0, 1, 1, 1, 0],
                        'IsActiveMember': [1, 1, 1, 0, 0],
                        'EstimatedSalary': [181534.04, 135502.77, 168261.4, 44937.01, 19150.14],
                        'Exited': [0, 0, 0, 1, 0],
                        'Geography_France': [0, 0, 0, 0, 0],
                        'Geography_Germany': [0, 0, 0, 1, 1],
                        'Geography_Spain': [1, 1, 1, 0, 0],
                        'Gender_Female': [0, 1, 1, 0, 1],
                        'Gender_Male': [1, 0, 0, 1, 0]
                        })
    balanced_data = sm_fit_resample(data)
    
    # check if the number of rows in the balanced data is equal to twice the number of the minority class (Exited=1)
    assert balanced_data.shape[0] == 2 * data[data['Exited'] == 1].shape[0]
    
    # check if the target variable (Exited) is balanced after oversampling
    assert balanced_data['Exited'].value_counts()[0] == balanced_data['Exited'].value_counts()[1]
    
    # check if the columns in the balanced data are the same as in the input data
    assert set(data.columns) == set(balanced_data.columns)
    
#########################
###### Scaling Data #####
#########################
@pytest.fixture
def data():
    data = pd.DataFrame({
        'CreditScore': [648, 693, 586, 438, 768],
        'Age': [55, 57, 33, 24, 43],
        'Tenure': [1, 9, 7, 8, 2],
        'Balance': [81370.07, 0.0, 0.0, 2312.43, 129264.05],
        'NumOfProducts': [1, 2, 2, 1, 2],
        'HasCrCard': [0, 1, 1, 1, 0],
        'IsActiveMember': [1, 1, 1, 0, 0],
        'EstimatedSalary': [181534.04, 135502.77, 168261.4, 44937.01, 19150.14],
        'Exited': [0, 0, 0, 1, 0],
        'Geography_France': [0, 0, 0, 0, 0],
        'Geography_Germany': [0, 0, 0, 1, 1],
        'Geography_Spain': [1, 1, 1, 0, 0],
        'Gender_Female': [0, 1, 1, 0, 1],
        'Gender_Male': [1, 0, 0, 1, 0]
    })
    return data

@pytest.fixture
def scaler(data):
    return fit_scaler(data)

def test_transform_data(data, scaler):
    columns_to_scale = ['CreditScore', 'Age', 'Tenure', 'EstimatedSalary', 'Balance']
    transformed_data = transform_data(data, scaler)
    assert not pd.isnull(transformed_data).any().any()
    assert (transformed_data[columns_to_scale].values.mean(axis=0) - 0.0 < 1e-6).all()
    assert (transformed_data[columns_to_scale].values.std(axis=0) - 1.0 < 1e-6).all()

def test_load_scaler(scaler):
    folder_path = 'model/5 - Model Final/'
    save_path = os.path.join(folder_path, 'scaler.pkl')
    with open(save_path, 'rb') as f:
        loaded_scaler = pickle.load(f)
    assert scaler.mean_.all() == loaded_scaler.mean_.all()
    assert scaler.var_.all() == loaded_scaler.var_.all()

def test_fit_scaler(data):
    columns_to_scale = ['CreditScore', 'Age', 'Tenure', 'EstimatedSalary', 'Balance']
    scaler = fit_scaler(data)
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_.shape[0] == len(data[columns_to_scale].columns)
    assert scaler.var_.shape[0] == len(data[columns_to_scale].columns)
#########################
##### Modeling Data #####
#########################
def test_binary_classification_xgb_tuned():
    # load example dataset
    config_data = load_config()
    X_sm_clean, y_sm =  load_smote_clean(config_data)
    X_valid_clean, y_valid = load_valid_clean(config_data)
    X_test_clean, y_test = load_test_clean(config_data)

    # call function to train and test the XGBoost model
    best_xgb_clf = binary_classification_xgb_tuned(x_train = X_sm_clean, y_train = y_sm, \
                                                   x_valid = X_valid_clean, y_valid = y_valid, \
                                                   x_test = X_test_clean, y_test = y_test)

    # check that the best_xgb_clf is an instance of XGBClassifier
    assert isinstance(best_xgb_clf, xgb.XGBClassifier)

    # check that the best_xgb_clf has been trained on the training set
    # assert 'booster' in best_xgb_clf.get_booster().get_dump()
    
    assert best_xgb_clf.score(X_sm_clean, y_sm) > 0
    
    train_acc = best_xgb_clf.score(X_sm_clean, y_sm)
    valid_acc = best_xgb_clf.score(X_valid_clean, y_valid)
    test_acc = best_xgb_clf.score(X_test_clean, y_test)

    assert train_acc > valid_acc
    assert train_acc > test_acc

    # check that the validation and test accuracies are between 0 and 1
    assert 0 <= valid_acc <= 1
    assert 0 <= test_acc <= 1
    
######################
###### SAVE LOG ######
######################
def test_save_model_log():
    # create sample data
    X_test = [[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]]
    y_test = [1, 0, 1]
    X_train = [[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9],
              [4, 7, 5],
              [8, 6, 4],
              [3, 8, 7],
              [5, 3, 5]]
    y_train = [0,0,1,1,0,1,0]
    
    # train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # call function
    logreg_model = save_model_log(model=model, model_name = "logreg", X_test = X_test, y_test=y_test)
    # menyimpan log sebagai file JSON
    with open('training_log/training_log_reg.json', 'w') as f:
        json.dump(logreg_model, f)
    
    # check if log file is created
    log_file_path = "training_log/training_log_reg.json"
    assert os.path.isfile(log_file_path), f"File {log_file_path} not found"











