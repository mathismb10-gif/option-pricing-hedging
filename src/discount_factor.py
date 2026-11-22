import numpy as np

def discount_factor(r,T): 
    return np.exp(-r*T)

