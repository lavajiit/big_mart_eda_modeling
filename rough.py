import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, Normalizer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import xgboost as xgb
import seaborn as sns
from imblearn.over_sampling import ADASYN
import os
import sys
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def training(X,y, model=LogisticRegression(), f_imp=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # fit the model
    model.fit(X_train, y_train)
    # Predict for training set:
    y_train_hat = model.predict(X_train)
    lloss_train = log_loss(y_train, y_train_hat)
    # Predict for test set:
    y_test_hat = model.predict(X_test)
    lloss_test = log_loss(y_test, y_test_hat)

    #Perform cross-validation:
    cv_score = cross_val_score(LogisticRegression(), X, y, cv=5, scoring='neg_log_loss')
    mean_CV = np.mean(cv_score)

    scores = pd.DataFrame({'':[lloss_train, lloss_test, mean_CV]}, index=['LLOSS_train','RMSLLOSS_test','LLOSS_CV'])

    if f_imp==True:
        importances(model, X_train.columns)
    return scores, lloss_train, lloss_test

def rank_models(X,y, classifiers, f_=False):

    scores_dict = {}
    for clf in classifiers:
        scores_dict[clf.__class__.__name__] = training(X,y, clf, f_imp=f_)[1:]
        
    table = pd.DataFrame.from_dict(scores_dict).T
    table.columns = ['LLOSS_train','LLOSS_test']
    
    return table.sort_values('LLOSS_test', ascending=False)

path = '/home/f/DS/Tasks/ThinkBumbleBee/Travel_Insurance_train.csv'
data = pd.read_csv(path)
X, y = data.drop('Claim', 1), data.Claim
train_X, test_X, train_y, test_y = train_test_split(X, y, stratify=y, test_size=0.2)
train_df = train_X[train_X.columns]
train_df['Claim'] = train_y

test_df = test_X[test_X.columns]
test_df['Claim'] = test_y

# del data, X, y, train_X, test_X, train_y, test_y

ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_resample(train_X, train_y)
print('Resampled dataset shape %s' % Counter(y_res))