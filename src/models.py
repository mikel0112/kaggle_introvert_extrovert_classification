import torch
import xgboost as xgb

class XGBoostModel:
    def __init__(self, model_type, num_classes, params):
        if model_type == 'classification':
            if num_classes == 2:
                self.model = xgb.XGBClassifier(
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
                self.model = xgb.XGBClassifier(
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
            self.model = xgb.XGBRegressor(
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