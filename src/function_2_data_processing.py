from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import copy
import joblib
import yaml

from datetime import datetime
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import datetime

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
        
############################################
def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_train = pickle_load(config_data["train_set_path"][0])
    y_train = pickle_load(config_data["train_set_path"][1])

    X_valid = pickle_load(config_data["valid_set_path"][0])
    y_valid = pickle_load(config_data["valid_set_path"][1])

    X_test = pickle_load(config_data["test_set_path"][0])
    y_test = pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat(
        [X_train, y_train],
        axis = 1
    )
    valid_set = pd.concat(
        [X_valid, y_valid],
        axis = 1
    )
    test_set = pd.concat(
        [X_test, y_test],
        axis = 1
    )

    # Return 3 set of data
    return train_set, valid_set, test_set

def load_data(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_train = pickle_load(config_data["train_set_path"][0])
    y_train = pickle_load(config_data["train_set_path"][1])

    X_valid = pickle_load(config_data["valid_set_path"][0])
    y_valid = pickle_load(config_data["valid_set_path"][1])

    X_test = pickle_load(config_data["test_set_path"][0])
    y_test = pickle_load(config_data["test_set_path"][1])

    # Return 3 set of data
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_data_scaling(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    X_sm_clean = pickle_load(config_data["standar_scaler_sm"][0])
    y_sm = pickle_load(config_data["standar_scaler_sm"][1])

    X_test_clean = pickle_load(config_data["standar_scaler_test"][0])
    y_test = pickle_load(config_data["standar_scaler_test"][1])

    X_valid_clean = pickle_load(config_data["standar_scaler_valid"][0])
    y_valid = pickle_load(config_data["standar_scaler_valid"][1])


def imputeData(data, numerical_columns_mean, numerical_columns_median, categorical_columns):
    """
    Fungsi untuk melakukan imputasi data numerik dan kategorikal
    :param data: <pandas dataframe> sample data input
    :param numerical_columns_mean: <list> list kolom numerik data yang akan diimputasi dengan mean
    :param numerical_columns_median: <list> list kolom numerik data yang akan diimputasi dengan median
    :param categorical_columns: <list> list kolom kategorikal data
    :return numerical_data_imputed: <pandas dataframe> data numerik imputed
    :return categorical_data_imputed: <pandas dataframe> data kategorikal imputed
    """
    # Imputasi kolom numerik dengan mean
    numerical_data_mean = data[numerical_columns_mean]
    imputer_numerical_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_numerical_mean.fit(numerical_data_mean)
    imputed_data_mean = imputer_numerical_mean.transform(numerical_data_mean)
    numerical_data_imputed_mean = pd.DataFrame(imputed_data_mean, columns=numerical_columns_mean, index=numerical_data_mean.index)

    # Imputasi kolom numerik dengan median
    numerical_data_median = data[numerical_columns_median]
    imputer_numerical_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer_numerical_median.fit(numerical_data_median)
    imputed_data_median = imputer_numerical_median.transform(numerical_data_median)
    numerical_data_imputed_median = pd.DataFrame(imputed_data_median, columns=numerical_columns_median, index=numerical_data_median.index)

    # Gabungkan kedua data numerik yang telah diimputasi
    numerical_data_imputed = pd.concat([numerical_data_imputed_mean, numerical_data_imputed_median], axis=1)

    # Seleksi data kategorikal
    categorical_data = data[categorical_columns]

    # Imputasi dengan menggunakan modus
    mode = categorical_data.mode().iloc[0]

    # Lakukan imputasi untuk data kategorikal
    categorical_data_imputed = categorical_data.fillna(mode)

    # Gabungkan data numerik dan kategorikal yang telah diimputasi
    data_imputed = pd.concat([numerical_data_imputed, categorical_data_imputed], axis=1)

    return data_imputed

def get_dummies(train_df, input_df):
    # Menggabungkan data train dan input menjadi satu DataFrame
    combined_df = pd.concat([train_df, input_df])
    
    # Mengubah variabel kategorikal menjadi variabel dummy
    dummies_df = pd.get_dummies(combined_df, columns=train_df.select_dtypes(include='object').columns)
    
    # Memisahkan kembali data train dan input
    train_dummies = dummies_df[:train_df.shape[0]]
    input_dummies = dummies_df[train_df.shape[0]:]
    
    return train_dummies, input_dummies

def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    sm = SMOTE(random_state = 112)

    # Balancing set data
    X_sm, y_sm = sm.fit_resample(
        set_data.drop("Exited", axis = 1),
        set_data.Exited
    )

    # Concatenate balanced data
    set_data_sm = pd.concat(
        [X_sm, y_sm],
        axis = 1
    )

    # Return balanced data
    return set_data_sm


columns_to_scale = ["CreditScore", "Age", "Tenure", "EstimatedSalary", "Balance"]

def fit_scaler(train_data):
    config_data = load_config()
    scaler = StandardScaler()
    scaler.fit(train_data.loc[:, columns_to_scale])
    # save scaler
    #with open('model/5 - Model Final/scaler.pkl', 'wb') as f:
    with open(config_data["model_scaler"], 'wb') as f:
        pickle.dump(scaler, f)
    return scaler

def load_scaler(folder_path):
    # load scaler
    file_path = os.path.join(folder_path, 'scaler.pkl')
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def transform_data(data, scaler):
    scaled_data = scaler.transform(data.loc[:, columns_to_scale])
    data.loc[:, columns_to_scale] = scaled_data
    return data

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = load_config()

    # 2. Load dataset
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data(config_data)
    
    #Imputasi Data
    #Imputasi Data
    numerical_column = ["Age"]
    numerical_column_mean = ["CreditScore", "Balance", "EstimatedSalary"]
    set_numerik = numerical_column + numerical_column_mean
    dataset_column = list(X_train.columns)
    categorical_column = list(set(dataset_column).difference(set(set_numerik)))
    
    X_train_impute = imputeData(data = X_train, 
                                numerical_columns_mean = numerical_column_mean, 
                                numerical_columns_median = numerical_column, 
                                categorical_columns = categorical_column)
    
    X_valid_impute = imputeData(data = X_valid, 
                                numerical_columns_mean = numerical_column_mean, 
                                numerical_columns_median = numerical_column, 
                                categorical_columns = categorical_column)
    
    X_test_impute = imputeData(data = X_test, 
                               numerical_columns_mean = numerical_column_mean, 
                               numerical_columns_median = numerical_column, 
                               categorical_columns = categorical_column)
    
    pickle_dump(X_train_impute, config_data["impute_data_train"][0])
    pickle_dump(y_train, config_data["impute_data_train"][1])
    pickle_dump(X_test_impute, config_data["impute_data_test"][0])
    pickle_dump(y_test, config_data["impute_data_test"][1])
    pickle_dump(X_valid_impute, config_data["impute_data_valid"][0])
    pickle_dump(y_valid, config_data["impute_data_valid"][1])
    
    #ohe
    dataset_ohe, valid_set = get_dummies(X_train_impute, X_valid_impute)
    dataset, test_set = get_dummies(X_train_impute, X_test_impute)

    # 12. SMOTE dataset
    dataset = pd.concat([dataset, y_train],axis = 1)
    dataset_smote = sm_fit_resample(dataset)
    #Fitting Scaler
    fitting_scaler = fit_scaler(dataset_smote)
    scaling = load_scaler(config_data["scaler"])
    # scaling = load_scaler('model/5 - Model Final/')
    # transform selected columns of training data
    
    dataset_clean = transform_data(dataset_smote, scaling)
    X_valid_clean = transform_data(valid_set, scaling)
    X_test_clean = transform_data(test_set, scaling)
    
    X_sm_clean = dataset_clean.drop(columns = "Exited")
    y_sm = dataset_clean["Exited"]
    
    X_sm_clean = X_sm_clean[sorted(X_sm_clean.columns)]
    X_valid_clean = X_valid_clean[sorted(X_valid_clean.columns)]
    X_test_clean = X_test_clean[sorted(X_test_clean.columns)]

    pickle_dump(X_sm_clean, config_data["standar_scaler_sm"][0])
    pickle_dump(y_sm, config_data["standar_scaler_sm"][1])
    
    pickle_dump(X_test_clean, config_data["standar_scaler_test"][0])
    pickle_dump(y_test, config_data["standar_scaler_test"][1])

    pickle_dump(X_valid_clean, config_data["standar_scaler_valid"][0])
    pickle_dump(y_valid, config_data["standar_scaler_valid"][1])
    