#!/usr/bin/env python
# coding: utf-8

# # Convergence of Binomial model and Trinomial model to the Black Scholes model

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Black scholes
import numpy as np
from scipy.stats import norm

N = norm.cdf

def BSM(S0, K, T, r, sigma,opttype):
    if opttype=='C':
        d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * N(d1) - K * np.exp(-r*T)* N(d2)
    elif opttype=='P':
        d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma* np.sqrt(T)
        return K*np.exp(-r*T)*N(-d2) - S0*N(-d1)
    else:
         raise ValueError("type_ must be 'C' or 'P'" )


# In[3]:


from math import comb
#Binomial
def Binomial(S0, K , T, r, sigma, n,opttype):
    dt = T/n
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (  np.exp(r*dt) - d )  /  (  u - d )
    value = 0 
    for i in range(n+1):
        node_prob = comb(n, i)*p**i*(1-p)**(n-i)
        ST = S0*(u)**i*(d)**(n-i)
        if opttype == 'C':
            value += max(ST-K,0) * node_prob
        elif opttype == 'P':
            value += max(K-ST, 0)*node_prob
        else:
            raise ValueError("type_ must be 'C' or 'P'" )
    
    return value*np.exp(-r*T)


# In[4]:


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


# In[5]:


# import matplotlib.pyplot as plt

# %matplotlib inline
# import numpy as np
# from qiskit import QuantumCircuit
# from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
# from qiskit.circuit.library import LinearAmplitudeFunction
# from qiskit_aer.primitives import Sampler
# from qiskit_finance.circuit.library import LogNormalDistribution
# from qiskit_finance.applications.estimation import EuropeanCallPricing


# num_uncertainty_qubits = 3
# # parameters for considered random distribution
# S = 100  # initial stock price
# vol = 0.2  # volatility of 20%
# r = 0.05  # annual interest rate of 5%
# T = 1  #1 year to maturity

# # resulting parameters for log-normal distribution
# mu = (r - 0.5 * vol**2) * T + np.log(S)
# sigma = vol * np.sqrt(T)
# mean = np.exp(mu + sigma**2 / 2)
# variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
# stddev = np.sqrt(variance)

# # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
# low = np.maximum(0, mean - 3 * stddev)
# high = mean + 3 * stddev

# # construct A operator for QAE for the payoff function by
# # composing the uncertainty model and the objective
# uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high))

# european_call_pricing = EuropeanCallPricing(
#     num_state_qubits=3,
#     strike_price=110,
#     rescaling_factor=0.01,
#     bounds=(low, high),
#     uncertainty_model=uncertainty_model,)
# # set target precision and confidence level
# epsilon = 0.01
# alpha = 0.05

# problem = european_call_pricing.to_estimation_problem()
# # construct amplitude estimation
# ae = IterativeAmplitudeEstimation(
#     epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 1000})
# )
# result = ae.estimate(problem)

# conf_int = np.array(result.confidence_interval_processed)
# #print("Exact value:        \t%.4f" % exact_value)
# print("Estimated value:    \t%.4f" % (european_call_pricing.interpret(result)))
# print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))


# In[6]:


# #q=[]
# q.append(european_call_pricing.interpret(result))
# print(q)


# In[26]:


#Parameters
S0 = 70      # initial stock price
K = 100       # strike price
T = 5        # time to maturity in years
r = 0.08      # annual risk-free rate
n = 100         # number of time steps
sigma=0.8       #volatility
opttype = 'P' # Option Type 'C' or 'P'

c=Binomial(S0, K, T, r,sigma, n,opttype)
b=BSM(S0,K,T,r,sigma,opttype)
t=Trinomial(S0, K, T, r, sigma, n,opttype)
print("binomial:",c)
print("Black Scholes:",b)
print("trinomial:",t)


# In[27]:


Ns = [5, 10,15, 20,25, 20, 35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150]
c=[] 
b=[]
t=[]
for n in Ns:
    c.append( Binomial(S0, K, T, r,sigma, n,opttype))
    b.append( BSM(S0,K,T,r,sigma,opttype))
    t.append(Trinomial(S0,K,T,r,sigma,n,opttype))


# In[28]:


plt.plot(Ns, c, label = "Binomial")
plt.plot(Ns, b, label = "Exact")
plt.plot(Ns, t, label = 'Trinomial')
plt.legend(loc='lower right')


# In[ ]:




