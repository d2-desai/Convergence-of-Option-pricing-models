#!/usr/bin/env python
# coding: utf-8

# # Trinomial Option Pricing Model
# 
# 

# In[49]:


#Trinomial
import math
def Trinomial(S0, K, T, r, sigma, n, opttype):
    if opttype=='C':
        C = {}
        dt = T/n
        Up = math.exp(sigma * math.sqrt(2*dt))
        Down = math.exp(-sigma * math.sqrt(2*dt))
        R = math.exp(r*dt)
        a = math.exp((r)*dt/2)
        a1 = math.exp(sigma * math.sqrt(dt/2))
        a2 = math.exp(-sigma * math.sqrt(dt/2))
        piUp = ((a - a2) / (a1 - a2))**2
        piDown = ((a1 - a) / (a1 - a2))**2
        pim = 1- piUp - piDown
        for m in range(0, 2*n+1):
                C[(n,m)] = max((S0*(Up**(max(m-n,0)))*(Down**(max(n*2-n-m,0))))- K, 0)
        for k in range(n-1, -1, -1):
                for m in range(0, 2*k+1):
                        C[(k,m)] = (piUp * C[(k+1, m+2)] + pim * C[(k+1, m+1)]+ piDown * C[(k+1, m)])/R 
        return C[(0,0)]
    elif opttype=='P':
        C = {}
        dt = T/n
        Up = math.exp(sigma * math.sqrt(2*dt))
        Down = math.exp(-sigma * math.sqrt(2*dt))
        R = math.exp(r*dt)
        a = math.exp((r)*dt/2)
        a1 = math.exp(sigma * math.sqrt(dt/2))
        a2 = math.exp(-sigma * math.sqrt(dt/2))
        piUp = ((a - a2) / (a1 - a2))**2
        piDown = ((a1 - a) / (a1 - a2))**2
        pim = 1- piUp - piDown
        for m in range(0, 2*n+1):
                C[(n,m)] = max(K - (S0*(Up**(max(m-n,0)))*(Down**(max(n*2-n-m,0)))), 0)
        for k in range(n-1, -1, -1):
                for m in range(0, 2*k+1):
                        C[(k,m)] = (piUp * C[(k+1, m+2)] + pim * C[(k+1, m+1)]+ piDown * C[(k+1, m)])/R 
        return C[(0,0)]


# In[57]:


# Example usage
S0 = 100
K = 110
r = 0.05
T = 1
sigma = 0.2
n = 100
opttype = 'C'
price = Trinomial(S0, K, T, r, sigma, n, opttype)

print("Option price:", price)


# In[56]:


S0 = 100
K = 110
r = 0.05
T = 1
sigma = 0.2
opttype = 'C'
Ns = [2, 4, 6, 8, 10, 20, 50, 100, 200, 300, 400]
    

for n in Ns:
    c = Trinomial(S, K, T, r, sigma, n, opttype)
    print(f'Price is {n} steps is {c}')


# In[ ]:





# In[ ]:




