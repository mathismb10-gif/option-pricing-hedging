import numpy as np

def call_payoff(S_T,K):
    return np.maximum(S_T-K,0)

def put_payoff(S_T,K):
    return np.maximum(K-S_T,0)
