import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import yaml
import pandas as pd
import csv
from data_preprocessing import DataPreprocessing
from train import load_data_plus_preprocessing, data_ready


if __name__ == '__main__':

    # load params
    params = yaml.safe_load(open('params.yaml', 'r'))

    # load data
    test_path = params['evaluation']['test_data_path']
    file_type = test_path.split('.')[-1]
    split_name = test_path.split('/')[-1].split('.')[0]

    if params['basics']['kaggle']:

        # load saved model
        model = joblib.load(f"outputs/training/{params['model']['name']}last_training/model.pkl")

        # same preprocessing as training
        data = load_data_plus_preprocessing(
            test_path, 
            file_type, 
            None,  # No target column in test data
            params['data_preprocessing']['sample_identificator_column']
        )
        last_training = params['training']['last_training']
        X_test = data_ready(data, 0, split_name, last_training)  # No split needed for test data
        # Evaluate model
        y_pred = model.model.predict(X_test)

        # convert predictions to classes if classification
        if params['basics']['problem_type'] == 'classification':
            y_pred = [params['basics']['classes'][i] for i in y_pred]
        # identifier
        sample_identifier = params['data_preprocessing']['sample_identificator_column']
        target_column = params['data_preprocessing']['target_column']
        predictions = pd.DataFrame({
            sample_identifier: data.index,
            target_column: y_pred
        })
        predictions.to_csv('outputs/predictions.csv', index=False)