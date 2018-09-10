
 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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
#print(df.head())


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
plt.savefig('2.png')
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
plt.savefig('1.png')
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
print("For Linear Regression")
classification_model(model, traindf, predictor_var, outcome_var)
print("")
print("")

# For only one Predictor
predictor_var = ['radius_mean']
model = LogisticRegression()
print("For Linear Regression With One Variable")
classification_model(model, traindf, predictor_var, outcome_var)
print("")
print("")

""" 
This gives a similar prediction accuracy and a cross-validation score.
The accuracy of the predictions are good but not great. The cross-validation 
scores are reasonable. 
"""

# Decision Trees
"""
Let’s consider a very basic example that uses data set for 
predicting whether a person has cancer or not. Below model uses 3 
features/attributes/columns from the data set, namely 'radius_mean',
'perimeter_mean','area_mean','compactness_mean' and 'concave points_mean'
"""

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
model = DecisionTreeClassifier()
print("For Decision Trees")
classification_model(model, traindf, predictor_var, outcome_var)
print("")
print("")

"""
Here we are  
"""

# For only single Predictor

predictor_var = ['radius_mean']
model = DecisionTreeClassifier()
print("For Decision Trees With One Variable")
classification_model(model, traindf, predictor_var, outcome_var)
print("")
print("")

"""
The accuracy of the prediction is much much better now. 

Using a single predictor gives a 97% prediction accuracy for this model but 
the cross-validation score is not that great.
"""

# Random Forest

predictor_var = features_mean
model = RandomForestClassifier(n_estimators = 100,min_samples_split = 25, 
                               max_depth = 7, max_features = 2)
print("For Random Forest")
classification_model(model, traindf,predictor_var, outcome_var)
print("")
print("")

"""
Using all the features improves the prediction accuracy and the cross-validation 
score is great.

An advantage with Random Forest is that it returns a feature importance 
matrix which can be used to select features. 
"""

# Selecting Top features 

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)

# Using Top 5 Features

predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean',]
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 25, 
                               max_depth = 7, max_features = 2)
classification_model(model, traindf, predictor_var, outcome_var)

"""
Using the top 5 features only changes the prediction accuracy a bit but
we get a better result if we use all the predictors.
"""


# Using on the test data set

#predictor_var = features_mean
model = RandomForestClassifier(n_estimators = 500,min_samples_split = 25,
                               max_depth = 7, max_features = 2)
classification_model(model, testdf, predictor_var, outcome_var)

"""
The prediction accuracy for the test data set using the above 
Random Forest model is 95%.
"""


"""
Conclusion
The best model to be used for diagnosing breast cancer as found in this 
analysis is the Random Forest model with the top 5 predictors, 
'concave points_mean','area_mean','radius_mean','perimeter_mean',
'concavity_mean'. It gives a prediction accuracy of ~95% and a 
cross-validation score ~ 93% for the test data set.
"""





























from sklearn import svm
clf = svm.SVC()
classification_model(clf, traindf, predictor_var, outcome_var)

















