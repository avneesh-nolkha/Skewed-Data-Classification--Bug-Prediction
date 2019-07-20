#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:04:53 2019

@author: avneeshnolkha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

#importing files

file1 = pd.read_csv("eclipse-metrics-files-2.0(1).csv",delimiter=';',error_bad_lines=False)
file2 = pd.read_csv("eclipse-metrics-files-3.0.csv",delimiter=';',error_bad_lines=False)

org_data=pd.DataFrame(pd.concat([file1,file2],axis=0))
org_data=org_data.dropna()

#Visualising distributin of pre release bugs
plt.figure(figsize=(10,8))
plt.title('Distribution of pre release bugs')
sns.countplot(org_data['pre'])

#if any software has post release bug, we will change it to true else flase
org_data['post']=org_data['post'] >0

#checking distribution of post release bugs
bug_count=org_data['post'].value_counts()
#visualising post release bigs
plt.figure(figsize=(10,8))
sns.barplot(x=bug_count.index, y=bug_count)
plt.title('Count of post release bugs')
plt.ylabel('Count')
plt.xlabel('Class (False:No bugs found, True:bugs found)')

no_bug= bug_count[0]
bug = bug_count[1]
perc_bug = (bug/(bug+no_bug))*100
perc_nobug = (nobug/(bug+no_bug))*100
print('There were {} bug free software release ({:.3f}%) and {} software releases with post release bugs({:.3f}%).'.format(no_bug, perc_nobug, bug, perc_bug))

"""As we can see, there is a large difference between the two classes. This is a skewed data classification problem.To deal with such kind of problem, we will make a seperate databse containing equal distribution of both classes. This will help our algorithm to distinguish better between the two classes and not get biased towards a single class.
Before creating subsample, let us pre-process and scale the data"""
dtype=org_data.dtypes
org_X = org_data.iloc[:,2:].drop('post',1).values
org_y = org_data.loc[:,'post'].values.ravel()
#Using column transformer for scaling data
colT = ColumnTransformer([("scaler",StandardScaler(),list(range(0,199)))])
org_X = colT.fit_transform(org_X)

"""Creating a subsample of data"""
#df with post release bugs
bugs=org_data[org_data['post']==True]
#df with no post release bugs
no_bugs=org_data[org_data['post']==False]
#for new df we need equal distribution of both classes
sub_df=no_bugs.sample(len(bugs))
sub_df.head()
#creating new df
bugs.reset_index(drop=True, inplace=True)
sub_df.reset_index(drop=True, inplace=True)
data = pd.DataFrame(pd.concat([bugs,sub_df],axis=0))
data=data.sample(frac=1).reset_index(drop=True)
data['post'].value_counts()
features=data.iloc[:,2:-1].drop('post',1).values
labels=data.loc[:,'post'].values.ravel()

x_train , x_test , y_train,y_test = train_test_split(features,labels,test_size=0.15,random_state=0)

#applying different algorithms
models = []

models.append(('Logistic', LogisticRegression()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('KNN Classifier', KNeighborsClassifier()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('Support Vector Classifier', SVC()))
models.append(('XG Boost Classifier', XGBClassifier()))
models.append(('Random Forest', RandomForestClassifier()))

#testing models
train_score = pd.DataFrame([])
for name, model in models:
    kfold = KFold(n_splits=10, random_state=42)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='recall')
    train_score = train_score.append(pd.DataFrame({'Model': name, 'Mean Recall_score': cv_results.mean(),'Recall_std':cv_results.std()}, index=[0]), ignore_index=True)
    


"""Now applying the algos on orignal dataset"""

org_features = org_data.iloc[:,2:-1].drop('post',axis=1).values
org_labels = org_data.loc[:,'post'].values.ravel()
test_score = pd.DataFrame([])
for name, model in models:
    model.fit(features,labels)
    labels_pred=model.predict(org_features)
    recall=recall_score(org_labels,labels_pred)
    test_score=test_score.append(pd.DataFrame({'Model':name,'Recall Score':recall},index=[0]), ignore_index=True)

    









































