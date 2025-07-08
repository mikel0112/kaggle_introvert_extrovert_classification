from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import json
import os
from data_preprocessing import DataPreprocessing, DataAnalysis
import datetime
import yaml
from xgboost import XGBClassifier, XGBRegressor
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

def data_ready(data, val_split):
    # Split data into train, validation
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)
    # concvert to numpy arrays
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    num_classes = len(np.unique(y))  # Get number of unique classes for XGBoost
    return X_train, X_val, y_train, y_val, num_classes

def xgboost_model(model_type, num_classes, params):
    if model_type == 'classification':
        if num_classes == 2:
            return XGBClassifier(
                objective=str(params['model']['hyperparameters']['objective']),
                eval_metric=str(params['model']['hyperparameters']['eval_metric']),
                use_label_encoder=params['model']['hyperparameters']['use_label_encoder'],
                random_state=params['model']['hyperparameters']['random_state'],
                n_estimators=params['model']['hyperparameters']['n_estimators'],
                max_depth=params['model']['hyperparameters']['max_depth'],
                learning_rate=params['model']['hyperparameters']['learning_rate'],
                colsample_bytree=params['model']['hyperparameters']['colsample_bytree'],
                subsample=params['model']['hyperparameters']['subsample'],
            )
        else:
            return XGBClassifier(
                objective=str(params['model']['hyperparameters']['objective']),
                eval_metric=str(params['model']['hyperparameters']['eval_metric']),
                use_label_encoder=params['model']['hyperparameters']['use_label_encoder'],
                random_state=params['model']['hyperparameters']['random_state'],
                n_estimators=params['model']['hyperparameters']['n_estimators'],
                max_depth=params['model']['hyperparameters']['max_depth'],
                learning_rate=params['model']['hyperparameters']['learning_rate'],
                colsample_bytree=params['model']['hyperparameters']['colsample_bytree'],
                subsample=params['model']['hyperparameters']['subsample'],
            )
    elif model_type == 'regression':
        return XGBRegressor(
                objective=str(params['model']['hyperparameters']['objective']),
                eval_metric=str(params['model']['hyperparameters']['eval_metric']),
                use_label_encoder=params['model']['hyperparameters']['use_label_encoder'],
                random_state=params['model']['hyperparameters']['random_state'],
                n_estimators=params['model']['hyperparameters']['n_estimators'],
                max_depth=params['model']['hyperparameters']['max_depth'],
                learning_rate=params['model']['hyperparameters']['learning_rate'],
                colsample_bytree=params['model']['hyperparameters']['colsample_bytree'],
                subsample=params['model']['hyperparameters']['subsample'],
            )
    else:
        raise ValueError("Unsupported model type. Please use 'classification' or 'regression'.")

if __name__ == '__main__':

    params = yaml.safe_load(open('params.yaml', 'r'))
    probl_type = params['basics']['problem_type']

    # Load data
    data_path = params['data_preprocessing']['data_path']
    file_type = data_path.split('.')[-1]
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
    X_train, X_val, y_train, y_val, num_classes = data_ready(data, val_split)
    
    # Define model
    if params['model']['name'] == 'xgboost':
        model = xgboost_model(probl_type, num_classes, params)
    
    # Train model
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        #early_stopping_rounds=params['training']['early_stopping_rounds'], 
        verbose=params['training']['verbose']
    )

    # graph showing training and validation loss
    os.makedirs(f'outputs/training/{params["model"]["name"]}', exist_ok=True)
    # make graph of training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(model.evals_result()['validation_0']['logloss'], label='Training Loss')
    plt.plot(model.evals_result()['validation_1']['logloss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/training/{params["model"]["name"]}/training_loss.png')
    plt.close()
    # Save model
    joblib.dump(model, f'outputs/training/{params["model"]["name"]}/model.pkl')