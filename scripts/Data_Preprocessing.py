#!/usr/bin/env python
# coding: utf-8

# #### Task 2: Data Preprocessing
# 
# Steps:
# - Handle missing values and outliers.
# - Encode categorical variables.
# - Normalize/standardize numerical features.
# - Split the data into training and testing sets.
# - Script: scripts/data_preprocessing.py

# In[1]:


#import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#load the data
#!pwd
datapath = "../data/boston_housing.csv"
datadf = pd.read_csv(datapath)

#display a few rows of the data
datadf.head()


# In[4]:


datadf.shape


# In[3]:


# Finding percentage outliers in every column

for k, v in datadf.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(datadf)[0]
        print("Column %s outliers = %.2f%%" % (k, perc))


# In[5]:


# removing outliers on target variable medv, refer to boxplots in EDA

datadf = datadf[~(datadf['medv'] >= 50)]


# In[6]:


datadf.shape


# In[11]:


#visualization of distribution of variables

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in datadf.items():
    sns.histplot(v, ax=axs[index],kde=True)
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# crim, zn, b are highly skewed, chas is categorical. medv is normal distributed

# In[13]:


# encode categorical variable chas
datadf['chas'].value_counts()


# The categorical variable `chas` is already encoded to 0 and 1 and type is `int`, there is no need for encoding

# In[20]:


# Normalize/standardize numerical features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
datadf_scaled = scaler.fit_transform(datadf) #returns numpy array
datadf_scaledf = pd.DataFrame(datadf_scaled, columns=datadf.columns)
datadf_scaledf.head()


# In[21]:


# pairwise correlation on data

plt.figure(figsize=(20, 10))
sns.heatmap(datadf_scaledf.corr().abs(),  annot=True)


# Findings on correlation:
# - variables with correlation score above 0.5 with target medv are good predictors
# - varibles for modeling at `indus, nox, rm, tax, ptratio and lstat`
# - `age` and `dis` have correlation below 0.5 but have useful information

# In[22]:


#features and target variables
X = datadf_scaledf[['indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'lstat']]       #features
y = datadf_scaledf['medv'] #target


# In[24]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=42)


# In[25]:


#Save output for modeling later

np.savez("train_test_split.npz",X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# In[ ]:




