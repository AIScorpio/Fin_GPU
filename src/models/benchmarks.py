import numpy as np
import time

# Black-Scholes
from scipy.stats import norm
def blackScholes(S0, K, r, sigma, T, opttype='P'):
    start = time.time()
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
    put_price = call_price - S0 + np.exp(-r*T)*K
    
    if opttype == 'C':
        # price = S0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
        price = call_price
    elif opttype == 'P':
        # price = np.exp(-r*T)*K*norm.cdf(-d2) - S0*norm.cdf(-d1) 
        price = put_price

    elapse = (time.time() - start) * 1e3
    print(f'Black-Scholes price: {price} - {elapse} ms')
    return price

# European stype via binomial tree
def binomialEuroOption(S0, K, r, sigma, nPeriod, T, opttype='P'):
    start = time.time()
    # precompute values
    dt = T / nPeriod                          # delta_t

    # for binomial tree 
    q = 0.0
    u = np.exp(sigma * np.sqrt(dt))           # up-factor in binomial tree
    d = 1/u                                   # down-factor in binomial tree, ensure recombining tree
    p = (np.exp((r - q) * dt) - d) / (u - d)  # probability 
    discount = np.exp(-r * dt)                # discount factor
    
    # initialize stock prices binomial matrix from time zero to maturity
    # number of nodes = N-1+2, indexing start from 0 to N, including N, in total N+1 nodes
    S = S0 * d**(np.arange(nPeriod, -1, -1)) * u**(np.arange(0, nPeriod+1, 1))
        
    # option payoffs at maturity
    if opttype=='P':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)

    # backward recursion thru the tree
    for i in np.arange(nPeriod-1, -1, -1):
        S = S0 * d**(np.arange(i, -1, -1)) * u**(np.arange(0, i+1, 1)) # np.array 自适应长度

        # 初始 C 为上一层, C[1:]为 up vector, C[:-1]为 down vector, 计算后 vector 长度 -1
        C = discount * ( p * C[1:] + (1-p) * C[:-1] )   
        # if opttype=='P':
        #     C = np.maximum(C, K - S)
        # else:
        #     C = np.maximum(C, S - K)

    elapse = (time.time() - start) * 1e3
    print(f'Binomial European price: {C[0]} - {elapse} ms')
    return C[0]


# American stype via binomial tree
def binomialAmericanOption(S0, K, r, sigma, nPeriod, T, opttype='P'):
    start = time.time()
    # precompute values
    dt = T / nPeriod                          # delta_t

    # for binomial tree 
    q = 0.0
    u = np.exp(sigma * np.sqrt(dt))           # up-factor in binomial tree
    d = 1/u                                   # down-factor in binomial tree, ensure recombining tree
    p = (np.exp((r - q) * dt) - d) / (u - d)  # probability 
    discount = np.exp(-r * dt)                # discount factor
    
    # initialize stock prices binomial matrix from time zero to maturity
    # number of nodes = N-1+2, indexing start from 0 to N, including N, in total N+1 nodes
    S = S0 * d**(np.arange(nPeriod, -1, -1)) * u**(np.arange(0, nPeriod+1, 1))
        
    # option payoffs at maturity
    if opttype=='P':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)

    # backward recursion thru the tree
    for i in np.arange(nPeriod-1, -1, -1):
        S = S0 * d**(np.arange(i, -1, -1)) * u**(np.arange(0, i+1, 1)) # np.array 自适应长度

        # 初始 C 为上一层, C[1:]为 up vector, C[:-1]为 down vector, 计算后 vector 长度 -1
        C = discount * ( p * C[1:] + (1-p) * C[:-1] )   
        if opttype=='P':
            C = np.maximum(C, K - S)
        else:
            C = np.maximum(C, S - K)
            
    elapse = (time.time() - start) * 1e3
    print(f'Binomial American price: {C[0]} - {elapse} ms')
    return C[0]
