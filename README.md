# kaggle_introvert_extrovert_classification
## Data
It is really important to have two different files one for training and another one for testing. And both file names contain train and test respectively (train.csv/xlsx and test.csv/xlsx)
## Preprocessing stage
It is divided into three main classes.
### 1. Data Cleaning
It  is inividual to every dataset. They final dataframe needs to have just numerical and categorical columns.
### 2. Data Preprocessing
### 3. Data Analysis
## Models
## Training stage
If working with xgboost first make a gridsearchcv and then based on the best params adjust, it is done using k-fold cross validation. For doing so, params, first training parameter. Then use best values to train the model and evaluate it, if it es enough, predict.
Try different models and settings.
## Evaluation stage
Evaluate every single model. Finally, with the best model [go to training stage](#training-stage) and using the whole training dataset without splitting it into train and val sets train again the best model and save it to use it for predictions (in params there is a parameter called last_training if it is ture the train data is not splitted).
Evaluations done in the three splits: train, val and test. Important to look at the best performance in test because of overtraining.
## Predictions stage
Only used for competitions like Kaggle so paramater Kaggle in params needs to be true. It makes a csv of the style needed to submit in the competition.