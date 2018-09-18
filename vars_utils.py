import numpy as np
from scipy.optimize import minimize_scalar  

def mean_diff(x, args):
    x_input, y = args[0], args[1]
    
    return ((np.abs(np.mean(y[x_input < x]) - np.mean(y[x_input >= x])) * -1.0) 
                + 
            np.abs((len(y[x_input < x]) * 100 / len(x_input)) - (len(y[x_input < x]) * 100 / len(x_input)) )
           )

def median_diff(x1, args):
    x_input, y = args[0], args[1]
    
    return ((np.abs(np.median(y[x_input < x]) - np.median(y[x_input >= x])) * -1.0) 
                + 
            np.abs((len(y[x_input < x]) * 100 / len(x_input)) - (len(y[x_input < x]) * 100 / len(x_input)) )
           )

'''
Given two numeric variables (one of them is the target variable),
it discretizes the non target variable so it maximizes the mean or median
difference between categories 
'''
def cut_variable_optimally(x, y, maximize='mean_diff', n_categories = 2):
    return minimize_scalar(mean_diff, 
                    bounds = [np.min(x), np.max(x)], 
                    args = [x, y], options = {'maxiter': 1000},
                    method = 'bounded')
    