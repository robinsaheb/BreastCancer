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
    
























