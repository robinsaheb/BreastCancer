#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:30:58 2017

@author: sahebsingh
"""


""" Load Libraries """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import seaborn as sns

""" Load the Data """

df = pd.read_csv('/Users/sahebsingh/Desktop/Projects/Data Mining/Dataset/data.csv', header = 0)
print(df.head())


""" Clean and Prepare the Data """

# Checking for null values.

def give_top_row(reader):
    topics = []
    for row in reader:
        topics.append(row)
    print(topics)
    
    for topic in topics:
        print("For Topic", topic, "there are",df[topic].isnull().sum(), "null values.")  
        print(' ')
        print(' ')

#give_top_row(df)


# Dropping values which gives no information.

df.drop('id', axis = 1, inplace = True)
df.drop('Unnamed: 32', axis = 1, inplace = True)

#give_top_row(df)

# Mapping M as 1 and B as 0.

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
print(df.describe())

# Visualisation of Diagnosis

plt.hist(df['diagnosis'])
plt.title('Diagnosis')
plt.show() # This shows that B is more than M.

# Diagnosis with respect to other variables

features_mean=list(df.columns[1:11])
# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]

plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()
    
""" Observations 
1. Mean values of cell radius, perimeter, area, compactness, concavity and 
concave points can be used in classification of the cancer.
2. mean values of texture, smoothness, symmetry or fractual dimension does
 not show a particular preference of one diagnosis over the other. In any
 of the histograms there are no noticeable large outliers that warrants 
 further cleanup.

"""


# Creating a test and training dataset.

traindf, testdf = train_test_split(df, test_size = 0.3)

# We are going to train different models to perform different model and
# evaluate it's performance.

def classification_model(model, data, predictors, outcome):
    
    # Fitting the Model
    model.fit(data[predictors], data[outcome])
    
    # Predicting the data
    predictions  = model.predict(data[predictors])
    
    # Printing The Accuracy 
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print('Accuracy: %s' %accuracy)
    
    # Using KFold for cross validation.
    kf = KFold(data.shape[0], n_folds = 5)
    error = []
    
    for train, test in kf:
        #Fitting the training data
        model.fit(data[predictors].iloc[train, :], data[outcome].iloc[train])
        
        # Predicting values using model
        predictions = model.predict(data[predictors].iloc[test, :])
        
        # Store the Cross Validation score to the error.
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
        
        print('Cross Validation of this model is %s' %np.mean(error))
    
# Logistic Regression
""" 
Logistic regression is widely used for classification of discrete data.
In this case we will use it for binary (1,0) classification.

Based on the observations in the histogram plots, we can see that diagnosis is
dependent on mean cell radius, mean perimeter, mean area, mean compactness, 
mean concavity and mean concave points. 
"""

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
outcome_var = 'diagnosis'
model = LogisticRegression()
classification_model(model, traindf, predictor_var, outcome_var)





















