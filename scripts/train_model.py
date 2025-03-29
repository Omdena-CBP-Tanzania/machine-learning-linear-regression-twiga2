#!/usr/bin/env python
# coding: utf-8

# #### Task 3: Model Training
# 
# Steps:
# - Choose appropriate features for the model.
# - Train a linear regression model.
# - Perform hyperparameter tuning (if applicable).
# 
# - Script: scripts/train_model.py

# In[2]:


#import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#load the preprocessed and split data

data = np.load("../data/train_test_split.npz")

X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']


# Features selection refer to `Data_Processing.ipynb`

# Training a linear regression model

# In[5]:


# Initialize the model
from sklearn.linear_model import LinearRegression

model = LinearRegression()


# In[6]:


# train the model
model.fit(X_train,y_train)


# In[7]:


#model coefficients

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")


# In[8]:


#save the model for evaluation later
from joblib import dump

dump(model,"../data/linearRegression.joblib")


# In[ ]:




