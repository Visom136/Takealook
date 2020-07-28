#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
sns.set()
from sklearn.linear_model import LinearRegression
from scipy import stats


# In[39]:


raw_data = pd.read_csv('downloads/houses_to_rent_v3.csv')
raw_data


# In[40]:


df = raw_data.copy()

df

df.head()


# In[52]:


y = df['total (R$)']
x1 = df['area']

y = y.values.reshape(-1,1)

x1 = x1.values.reshape(-1,1)

x1.shape

y.shape 

reg = LinearRegression()

reg.fit(x1,y)

reg.score(x1,y)

reg.intercept_


# In[53]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[54]:


plt.scatter(df['area'],df['total (R$)'])
yhat = reg.coef_ * x + reg.intercept_
fig = plt.plot(x,yhat, lw=3, c='red', label ='regression line')
plt.xlabel('area', fontsize = 20)
plt.ylabel('total (R$)', fontsize = 20)
plt.show()


# In[55]:


q_low = df['total (R$)'].quantile(0.01)
q_hi  = df['total (R$)'].quantile(0.99) 

df_filtered = df[(df['total (R$)'] < q_hi) & (df['total (R$)'] > q_low)]


# In[56]:


q1_low = df['area'].quantile(0.01)
q1_hi = df['area'].quantile(0.99)

df_filtered1 = df[(df['area'] < q1_hi) & (df['area'] > q1_low)]


# In[57]:


df_filtered2 = df_filtered + df_filtered1


# In[58]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[61]:


plt.scatter(df_filtered2['area'], df_filtered2['total (R$)'])
yhat = reg.coef_ * x + reg.intercept_
df_filtered2['area'] = df_filtered2['area'].values.reshape(-1,1)
fig = plt.plot(df_filtered2['area'],yhat, lw = 4, c='red', label='regression line')
plt.xlabel('area', fontsize = 20)
plt.ylabel('total (R$)', fontsize = 20)
plt.show()


# In[ ]:




