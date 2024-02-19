#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from sklearn import tree
from sklearn import ensemble
import warnings
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import catboost as cb
from sklearn import preprocessing

df=pd.read_csv('Traffic.csv')
df.drop(columns=['Time', 'Day of the week', 'Traffic Situation'], axis=1, inplace=True)
df.describe()


# In[2]:


x_train, x_test, y_train, y_test = train_test_split(df.loc[:, 'CarCount': 'BikeCount' ], df['BusCount'], train_size=0.8, random_state=271)


# In[3]:


train = x_train.copy()
train['BusCount'] = y_train


# In[4]:


datasets_amount = 10
interval = len(train)//datasets_amount
from_ = 0
to_ = interval
datasets = []
for i in range(datasets_amount):
    datasets.append(train.iloc[from_:to_, :])
    from_ += interval
    to_ += interval


# In[5]:


max_depth = 20
models = []
for ds in datasets:
    model_tree = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    model_tree.fit(train.drop('BusCount', axis=1), train['BusCount'])
    models.append(model_tree)


# In[6]:


y_pred = []
for model in models:
    y_predict = model.predict(x_test)
    y_pred.append(y_predict)
    print('метрика:')
    print(classification_report(y_test, y_predict))
y_pred


# In[7]:


mean_pred = np.array(y_pred).mean(axis=0)
mean_pred


# In[8]:


print(classification_report(y_test, mean_pred))


# In[9]:


model = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=1)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))


# In[10]:


get_ipython().run_cell_magic('time', '', "random_forest = RandomForestClassifier()\nparam_grid = {\n    'max_depth': [12, 18],\n    'min_samples_leaf': [3, 10],\n    'min_samples_split': [6, 12]\n}\ngrid_search_random_forest = GridSearchCV(estimator=random_forest,\n                                         param_grid=param_grid,\n                                         scoring='f1_macro',\n                                         cv=4)\ngrid_search_random_forest.fit(x_train, y_train)\nbest_model = grid_search_random_forest.best_estimator_\nbest_model\n")


# In[11]:


y_predict = best_model.predict(x_test)
print(classification_report(y_test, y_predict))


# In[12]:


get_ipython().run_cell_magic('time', '', 'model_catboost_clf = cb.CatBoostClassifier(iterations=1000)\nmodel_catboost_clf.fit(x_train, y_train)\ny_predict = model_catboost_clf.predict(x_test)\n')


# In[13]:


print(classification_report(y_test, y_predict))







