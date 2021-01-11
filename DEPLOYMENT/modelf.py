# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:11:10 2020

@author: Yugesh
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#LOAD THE DATASET
train_data = pd.read_csv("train1e.csv")
test_data = pd.read_csv("test1e.csv")

#droping ['id'] column from train data
train_data.drop(["Unnamed: 0"], axis = 1, inplace = True)

#droping ['id'] column from train data
test_data.drop(["Unnamed: 0"], axis = 1, inplace = True)


X = train_data.iloc[:, 0:4].values  
y = train_data.iloc[:, 4].values  


from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()

model=rf.fit(X_train, y_train)
y_train=pd.DataFrame(y_train)

from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor

rf_multioutput = MultiOutputRegressor(ensemble.RandomForestRegressor())

rf_multioutput.fit(X_train, y_train)
# Saving model to disk
pickle.dump(rf_multioutput, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6,10]]))
