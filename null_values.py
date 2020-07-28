# -*- coding: utf-8 -*-
"""null_values.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bGwGipPv1EZ-NmAPdUHzF6-GC3JMHLni
"""

import pandas as pd
from sklearn.impute import SimpleImputer

def null_values(train, test, t):

  """ input: train and test sets after unbalanced_classes
  output: y_train, X_train, y_test, X_test 
  *if a feature has more than t% of null values, drop it """
  
  # split the dataset
  y_train = train["class"]
  X_train = train.drop("class", axis=1)

  y_test = test["class"]
  X_test = test.drop("class", axis=1)

  # drop the features with more than t% of null values on the train set
  X_train = X_train.loc[:, X_train.isnull().mean() <= t]
  # remove those columns from the test set
  X_test = X_test[X_train.columns]

  # fill the remaining null values with the median of the corresponding feature
  imp = SimpleImputer(strategy = "median")
  T = pd.DataFrame(imp.fit_transform(X_train))
  T.columns = X_train.columns
  T.index = X_train.index
  X_train = T

  T = pd.DataFrame(imp.fit_transform(X_test))
  T.columns = X_test.columns
  T.index = X_test.index
  X_test = T
  
  return y_train, X_train, y_test, X_test