import numpy as np
from src.discount_factor import discount_factor

def put_from_call(C,S0,K,r,T):
    return C + K*discount_factor(r,T) - S0

def call_from_put(P,S0,K,r,T):
    return S0 + P - K*discount_factor(r,T)

def parity_gap(C,P,S0,K,r,T):
    return C-P+K*discount_factor(r,T)-S0



parity_gap(5, 4, 100, 100, 0.05, 1)
