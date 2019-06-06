import pandas as pd
import numpy as np
np.random.seed(24)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

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

# segregate training and test data
data_train = data.loc[data.source=='train']
data_test = data.loc[data.source=='test']




# Fit the model on the data
X = data_train.drop(['Item_Outlet_Sales', 'source'],1)
y = data_train['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression().fit(X_train, y_train)

# Predict training set:
y_train_hat = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_hat))
# Predict test set:
y_test_hat = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_hat))

#Perform cross-validation:
cv_score = cross_val_score(LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error')
mean_CV = np.sqrt(np.abs(np.mean(cv_score)))

scores = pd.DataFrame({'':[rmse_train, rmse_test, mean_CV]}, index=['RMSE_train','RMSE_test','RMSE_CV'])

print(scores)