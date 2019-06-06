import pandas as pd
import numpy as np
np.random.seed(24)
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

os.chdir('/home/petrichor/DS/Analytics_Vidhya_24_projects/3_Bigmart_Sales/')
train = pd.read_csv('Train_UWu5bXk.csv')
test = pd.read_csv('Test_u94Q5KV.csv')
submission = pd.read_csv('SampleSubmission_TmnO39y.csv')

# comibne train and test data
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort=False)

# MVT
data['Item_Weight'] = data.groupby(['Item_Identifier'])['Item_Weight'].transform(lambda x: x.fillna(x.mean()))
data['Outlet_Size'] = data['Outlet_Size'].fillna('Small')

# MVT + FE
data['Item_Visibility'] = data['Item_Visibility'].replace(0, np.nan)
data['Item_Visibility'] = data.groupby(['Item_Identifier'])['Item_Visibility'].transform(lambda x: x.fillna(x.mean()))

# FE
# Get the first two characters of ID:
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(
    {
        'low fat': 'Low Fat',
        'LF':'Low Fat',
        'reg':'Regular'})

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Outlet_Age'] = 2013 - data['Outlet_Establishment_Year']
data = data.drop(['Item_Type','Item_Identifier','Outlet_Establishment_Year'],1)

# EN
data = pd.get_dummies(data, columns=['Item_Fat_Content','Item_Type_Combined','Outlet_Identifier','Outlet_Size',
'Outlet_Location_Type','Outlet_Type'])

# SCALE
# for col in data.select_dtypes(include=[np.int]).columns:
#     data[col] = data[col].astype(float)

# scaler = StandardScaler()
# data[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Age']] = scaler.fit_transform(data[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Age']])

# segregate training and test data
data_train = data.loc[data.source=='train']
data_test = data.loc[data.source=='test']

target='Item_Outlet_Sales'
predictors = [x for x in data_train.columns if x not in [target]+['source']]

X = data_train[predictors]
y = data_train[target]

# use CV to find n_estimators
dtrain = xgb.DMatrix(X.values, label=y.values)
param = {'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'reg:linear', 'colsample_bytree':0.8, 'min_child_weight':1, 'gamma':0, 'subsample':0.8, 'scale_pos_weight':1}
cv_score = xgb.cv(param, dtrain, num_boost_round=200, nfold=5, metrics='rmse', early_stopping_rounds=5, verbose_eval=None)
# get n_estimators from cv_score
nbr = cv_score.shape[0]
lr = 0.1

# ------------  GRID SEARCH  ------------
grid = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
}
# max_depth, min_child_weight need not be included in xgb.XGBRegressor()
model = xgb.XGBRegressor(n_estimators=nbr, learning_rate=lr, silent=1, objective='reg:linear', colsample_bytree=0.8, gamma=0, subsample=0.8, scale_pos_weight=1)

gs = GridSearchCV(estimator=model, param_grid=grid, scoring=-1*((-1*'neg_mean_squared_error')**0.5), n_jobs=4, iid=False, cv=5, return_train_score=False)

gs.fit(X,y)
print(gs.cv_results_, gs.best_params_, gs.best_score_)
