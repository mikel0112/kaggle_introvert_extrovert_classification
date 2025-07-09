import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import json
import yaml
import pandas as pd
import csv
from data_preprocessing import DataPreprocessing
from train import load_data_plus_preprocessing, data_ready
import os

if __name__ == '__main__':

    # load params
    params = yaml.safe_load(open('params.yaml', 'r'))

    # load data
    data_path = params['data_preprocessing']['data_path']
    file_type = data_path.split('.')[-1]
    split_name = data_path.split('/')[-1].split('.')[0]
    target_column = params['data_preprocessing']['target_column']
    sample_identifier = params['data_preprocessing']['sample_identificator_column']
    val_split = params['training']['validation_split']
    train_val_data = load_data_plus_preprocessing(
        data_path, 
        file_type, 
        target_column, 
        sample_identifier, 
    )
    # divide data into train, validation, and test sets
    X_train, X_val, y_train, y_val, num_classes = data_ready(train_val_data, val_split, split_name, False, target_column)
    # make validation dataframe from x_val and y_val
    data_val = pd.DataFrame(X_val, columns=train_val_data.columns.drop(target_column))
    data_val[target_column] = y_val
    X_val, X_test, y_val, y_test, _ = data_ready(data_val, 0.5, split_name, False, target_column)
    
    # load saved model
    model = joblib.load(f"outputs/training/{params['model']['name']}/model.pkl")
    # make predictions in every split
    y_train_pred = model.model.predict(X_train)
    y_val_pred = model.model.predict(X_val)
    y_test_pred = model.model.predict(X_test)

    # save metrics for train, validation, and test sets
    metrics = {
        'train': {},
        'validation': {},
        'test': {}
    }
    problem_type = params['basics']['problem_type']
    if problem_type == 'classification':
        # train metrics
        metrics['train']['accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['train']['precision'] = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['train']['recall'] = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['train']['f1_score'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['train']['confusion_matrix'] = confusion_matrix(y_train, y_train_pred).tolist()
        # validation metrics
        metrics['validation']['accuracy'] = accuracy_score(y_val, y_val_pred)
        metrics['validation']['precision'] = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)        
        metrics['validation']['recall'] = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        metrics['validation']['f1_score'] = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        metrics['validation']['confusion_matrix'] = confusion_matrix(y_val, y_val_pred).tolist()
        # test metrics
        metrics['test']['accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['test']['precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)        
        metrics['test']['recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['test']['f1_score'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['test']['confusion_matrix'] = confusion_matrix(y_test, y_test_pred).tolist()
    elif problem_type == 'regression':
        # train metrics
        metrics['train']['r2_score'] = r2_score(y_train, y_train_pred)
        metrics['train']['mean_absolute_error'] = mean_absolute_error(y_train, y_train_pred)
        metrics['train']['root_mean_squared_error'] = root_mean_squared_error(y_train, y_train_pred)
        # validation metrics
        metrics['validation']['r2_score'] = r2_score(y_val, y_val_pred)
        metrics['validation']['mean_absolute_error'] = mean_absolute_error(y_val, y_val_pred)
        metrics['validation']['root_mean_squared_error'] = root_mean_squared_error(y_val, y_val_pred)
        # test metrics
        metrics['test']['r2_score'] = r2_score(y_test, y_test_pred)        
        metrics['test']['mean_absolute_error'] = mean_absolute_error(y_test, y_test_pred)
        metrics['test']['root_mean_squared_error'] = root_mean_squared_error(y_test, y_test_pred)
    os.makedirs(f"outputs/evaluation/{params['model']['name']}", exist_ok=True)
    with open(f"outputs/evaluation/{params['model']['name']}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)