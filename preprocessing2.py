#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import pandas as pd


# In[22]:


dataset = pd.read_excel('datase.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[23]:


print(X)


# In[24]:


print(y)


# In[25]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[26]:


print(X)


# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X= np.array(ct.fit_transform(X))


# In[28]:


print(X)


# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[30]:


print(y)


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, text_size = 0.2, random_state = 1)


# In[ ]:




