import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import yaml


if __name__ == '__main__':

    # load params
    params = yaml.safe_load(open('params.yaml', 'r'))

    # load data
    X_test = params['data']['test_data']
    # añadir lo de target column a y_test si existe

    # make preprocessing as in training

    # load saved model
    model = joblib.load(f'outputs/training/{params['model']['name']}/model.pkl')

    # Evaluate model
    y_pred = model.predict(X_test)

    if y_test is not None:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        with open('outputs/model/metrics_2.json', 'w') as f:
            json.dump(metrics, f, indent=4)