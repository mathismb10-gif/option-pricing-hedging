from src.discount_factor import discount_factor
import numpy as np 

def call_bounds(S0,K,r,T): 
    return np.maximum(S0 - K*discount_factor(r,T),0),S0 