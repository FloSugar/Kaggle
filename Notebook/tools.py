# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:59:14 2015

@author: Flogeek
"""
import numpy as np
import pandas as pd

from operator import itemgetter

from sklearn.base import TransformerMixin

#### Preprocessing function

### Missing values
#imputation

class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object ('O') are imputed with the most frequent value 
        in column.
    
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[col].value_counts().index[0]
            if X[col].dtype == np.dtype('O') else X[col].median() for col in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
        
### Features function
# get dummies, concat and delete feature

def feature_to_dummy(df, feature):
    ''' take a feature from a dataframe, convert it to dummy and name it like feature_value'''
    tmp = pd.get_dummies(df[feature], prefix=feature, prefix_sep='_')
    df = pd.concat([df, tmp], axis=1, join_axes=[df.index])
    del df[feature]
    return df
    
    

#### Scikit function
#grid_search

def report_grid(grid_scores, n_top=3):
    '''return the 3 best models : score & std(score) on CV samples'''
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
