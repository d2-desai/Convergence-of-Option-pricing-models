#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math as m


# In[2]:


#Parameters
S0 = 50      # initial stock price
K = 50       # strike price
T = 1      # time to maturity in years
r = 0.25      # annual risk-free rate
Su = 100      # up-factor in binomial models
Sd = 25     # ensure recombining tree


# In[3]:


u=Su/S0
d=Sd/S0


# In[4]:


q=((pow(m.e,r*T))-d)/(u-d)


# In[5]:


f_u=max(u*S0-K,0)


# In[6]:


f_d=max(d*S0-K,0)


# In[7]:


Op=(q*f_u+(1-q)*f_d)/(pow(m.e,r*T))
print(Op)

