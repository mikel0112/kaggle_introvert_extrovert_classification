from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import json
import os
from data_preprocessing import DataPreprocessing, DataAnalysis
import datetime
import yaml
from models import XGBoostModel
import matplotlib.pyplot as plt

def load_data_plus_preprocessing(
        file_path, 
        file_type, 
        target_column, 
        sample_identifier=None,
    ):
    # Load data
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'xlsx' or file_type == 'xls':
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'xlsx'.")
    
    ######### START Data Preprocessing #########

    dp = DataPreprocessing(df, target_column, sample_identifier)
    data_df = dp.convert_to_index()
    # Normalize and one hot encode data
    data_df = dp.normalize_and_one_hot_encode()
    # Convert to tensor
    #data_dict = dp.convert_to_list(data_dict)

    ######### END Data Preprocessing #########
    
    if target_column is not None:
        ######### START Data Analysis #########

        # create a report with the data analysis outputs
        da = DataAnalysis(data_df, target_column)
        os.makedirs('outputs/DataAnalysis', exist_ok=True)
        # Create correlation matrix as heatmap
        da.correlation_matrix()
        # Create correlation of 'Form 1-10' with other columns
        da.graph_target_correlation()
        # Create PCA and TSNE graphs
        #da.pca_calc_tsne_graph()

        ######### END Data Analysis #########

    return data_df

def data_ready(data, val_split, split_name, last_training, target_column):
    if split_name == 'train':
        # Split data into train, validation
        X = data.drop(columns=[target_column])
        y = data[target_column]
        num_classes = len(np.unique(y))  # Get number of unique classes for XGBoost
        if not last_training:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)
            # concvert to numpy arrays
            X_train = X_train.to_numpy()
            X_val = X_val.to_numpy()
            y_train = y_train.to_numpy()
            y_val = y_val.to_numpy()
            return X_train, X_val, y_train, y_val, num_classes
        else:
            # If last training, use all data for training and no validation
            X_train = X.to_numpy()
            y_train = y.to_numpy()
            return X_train, y_train, num_classes
    elif split_name == 'test':
        # convert data to numpy arrays
        X_test = data.to_numpy()
        return X_test


if __name__ == '__main__':

    params = yaml.safe_load(open('params.yaml', 'r'))
    probl_type = params['basics']['problem_type']

    # Load data
    data_path = params['data_preprocessing']['data_path']
    file_type = data_path.split('.')[-1]
    split_name = data_path.split('/')[-1].split('.')[0]
    target_column = params['data_preprocessing']['target_column']
    sample_identifier = params['data_preprocessing']['sample_identificator_column']
    val_split = params['training']['validation_split']
    data = load_data_plus_preprocessing(
        data_path, 
        file_type, 
        target_column, 
        sample_identifier, 
    )

    # get data ready for training
    last_training = params['training']['last_training']
    if last_training:
        last = 'last_training'
        X_train, y_train, num_classes = data_ready(data, val_split, split_name, last_training, target_column)
    else:
        last = None
        X_train, X_val, y_train, y_val, num_classes = data_ready(data, val_split, split_name, last_training, target_column)
        # make validation dataframe from x_val and y_val
        data_val = pd.DataFrame(X_val, columns=data.columns.drop(target_column))
        data_val[target_column] = y_val
        X_val, X_test, y_val, y_test, _ = data_ready(data_val, 0.5, split_name, False, target_column)
    # Define model
    if params['model']['name'] == 'xgboost':
        model = XGBoostModel(
            probl_type, 
            num_classes, 
            params
        )
    
    # Train model
    if last_training:
        print("Training model with all data without validation split.")
        model.model.fit(
            X_train, 
            y_train, 
            eval_set=None, 
            verbose=params['training']['verbose']
        )
    else:
        print("Training model with validation split.")
        # Fit model with training and validation data
        model.model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            #early_stopping_rounds=params['training']['early_stopping_rounds'], 
            verbose=params['training']['verbose']
        )

    # graph showing training and validation loss
    if not last_training:
        os.makedirs(f'outputs/training/{params["model"]["name"]}', exist_ok=True)
        # make graph of training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(model.model.evals_result()['validation_0']['logloss'], label='Training Loss')
        plt.plot(model.model.evals_result()['validation_1']['logloss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'outputs/training/{params["model"]["name"]}/training_loss.png')
        plt.close()
        # save model
        joblib.dump(model, f'outputs/training/{params["model"]["name"]}/model.pkl')
    else:
        os.makedirs(f'outputs/training/{params["model"]["name"]}{last}', exist_ok=True)
        # Save model
        joblib.dump(model, f'outputs/training/{params["model"]["name"]}{last}/model.pkl')