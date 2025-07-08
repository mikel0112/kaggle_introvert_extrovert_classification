import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import datetime
from torch.nn import functional as F
import torch
from albumentations import transforms
import random
import json

"""
The only CLASS that needs to be specific for each dataset is the DataCleaning class.
The rest of the classes can be used for any dataset.
"""

# Class to clean the data
class DataCleaning:
    def __init__(self, df):
        self.df = df

    def drop_empty_rows_and_columns(self):
        # drop empty rows
        self.df = self.df.dropna(how='all')
        # drop empty columns
        self.df = self.df.dropna(axis=1, how='all')
        return self.df
    
    def change_column_types(self, cols_list):
        
        return self.df
    
    def handle_categorical_data(self):
        for col in self.df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
        return self.df

# Class to preprocess the data, get ready for the model
class DataPreprocessing:
    def __init__(self, df, target_column, sample_identifier=None):
        self.df = df
        self.target_column = target_column
        self.sample_identifier = sample_identifier
    # convert identifier column to index
    def convert_to_index(self):
        if self.sample_identifier is not None and self.sample_identifier in self.df.columns:
            # set sample identifier as index
            self.df.set_index(self.sample_identifier, inplace=True)
        else:
            # if sample identifier is not provided or not in the dataframe, use default index
            self.df.reset_index(drop=True, inplace=True)
        return self.df
    # normalize numerical columns and one hot encode categorical columns
    def normalize_and_one_hot_encode(self):
        for col in self.df.select_dtypes(include=['float', 'int']).columns:
            # normalize numerical columns
            self.df[col] = (self.df[col] - self.df[col].mean()) / (self.df[col].std() + 1e-8)  # Adding a small value to avoid division by zero
        # one hot encode categorical columns
        for col in self.df.select_dtypes(include=['object']).columns:
            if col != self.target_column:
                one_hot = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, one_hot], axis=1)
                self.df = self.df.drop(columns=[col])
        # convert target column to numerical in test set it is not necessary
        if self.target_column in self.df.columns:
            le = LabelEncoder()
            self.df[self.target_column] = le.fit_transform(self.df[self.target_column])
        return self.df

# Class to analyze the data
class DataAnalysis:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def correlation_matrix(self):
        corr = self.df.corr()
        plt.figure(figsize=(48, 40))
        plt.title('Correlation Matrix', fontsize=40)
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        os.makedirs('outputs/DataAnalysis', exist_ok=True)
        plt.savefig('outputs/DataAnalysis/correlation_matrix.png')
        plt.close()
    
    def graph_target_correlation(self):
        corr = self.df.corr()
        # make a table with the correlation values of the target_column column
        corr_table = corr[self.target_column]
        corr_table = corr_table.sort_values(ascending=False)
        # save as image
        plt.figure(figsize=(12, 8))
        plt.title('Correlation of Form 1-10 with other columns')
        sns.barplot(x=corr_table.index, y=corr_table.values)
        # show values
        for i in range(len(corr_table)):
            plt.text(i, corr_table[i], round(corr_table[i], 2), ha='center')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('outputs/DataAnalysis/form_correlation.png')
        plt.close()
    
    def pca_calc_tsne_graph(self):
        # PCA one for every class we have 10
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, random_state=42)
        pca_results = []
        tsne_results = []
        # unique values of target_column
        unique_values = self.df[self.target_column].unique().tolist()
        for i, value in enumerate(unique_values):
            df_class = self.df[self.df[self.target_column] == value]
            df_class = df_class.drop(columns=[self.target_column])
            class_pca = pca.fit_transform(df_class)
            pca_results.append(class_pca)
            class_tsne = tsne.fit_transform(df_class)
            tsne_results.append(class_tsne)
        # plot pca
        plt.figure(figsize=(12, 8))
        for i in range(len(unique_values)):
            plt.scatter(pca_results[i][:, 0], pca_results[i][:, 1], label=f'Class {i}')
        plt.title('PCA')
        plt.legend()
        plt.savefig('outputs/DataAnalysis/pca.png')
        plt.close()
        
        # plot tsne
        plt.figure(figsize=(12, 8))
        for i in range(len(unique_values)):
            plt.scatter(tsne_results[i][:, 0], tsne_results[i][:, 1], label=f'Class {i}')
        plt.title('TSNE')
        plt.legend()
        plt.savefig('outputs/DataAnalysis/tsne.png')
        plt.close()
        


