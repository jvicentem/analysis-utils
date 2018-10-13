import numba
import numpy as np
from scipy.optimize import minimize_scalar  
from scipy.optimize import differential_evolution
import sys
import pyximport; pyximport.install()
import warnings

@numba.jit(nopython=True)
def mean_diff(x, args):
    x_input, y = args[0], args[1]
    
    return ((np.abs(np.mean(y[x_input < x]) - np.mean(y[x_input >= x])) * -1.0) 
                + 
            np.abs((len(y[x_input < x]) * 100 / len(x_input)) - (len(y[x_input >= x]) * 100 / len(x_input)) )
           )

@numba.jit(nopython=True)
def median_diff(x, args):
    x_input, y = args[0], args[1]

    first_half = y[x_input < x]
    second_half = y[x_input >= x]

    x_input_len = len(x_input)
    
    return ((np.abs(np.median(first_half) - np.median(second_half)) * -1.0) 
                + 
            np.abs((len(first_half) * 100 / x_input_len) - (len(second_half) * 100 / x_input_len) )
           )

'''
Given two numeric variables (one of them is the target variable),
it discretizes the non target variable so it maximizes the mean or median
difference between categories 
'''
@numba.jit(nopython=True)
def cut_variable_optimally(x, y, maximize='mean_diff', n_categories = 2):
    min_x = np.min(x)
    max_x = np.max(x)

    return minimize_scalar(mean_diff, bounds = [min_x, max_x], 
                            args = [x, y], options = {'maxiter': 1000},
                            method = 'bounded')

def median_diff_gen(x, args):   
    x_len = len(x)

    segments = []

    if len(args) > 1: 
        x_input, y, unbalanced_cuts_importance = args[0], args[1], args[2]
    else:
        x_input, y, unbalanced_cuts_importance = args[0][0], args[0][1], args[0][2]

    x_input_len = len(x_input)

    # First segment
    segment_one_elements = y[x_input < x[0]]
    segment_one_elements_len = len(segment_one_elements)
    segment_one_elements_ratio = segment_one_elements_len / x_input_len
    if (segment_one_elements_len > 0):
        segment_one_median = np.median(segment_one_elements)
    else:
        return sys.maxsize

    segments.append({'segment_elements': segment_one_elements, 'segment_elements_len': segment_one_elements_len, 
                     'segment_elements_ratio': segment_one_elements_ratio, 'segment_median': segment_one_median})

    # Middle segments
    for i in range(1, (x_len-1)):
        segment_middle_elements = y[(x_input >= x[i]) & (x_input < x[i+1])]
        segment_middle_elements_len = len(segment_middle_elements)
        segment_middle_elements_ratio = segment_middle_elements_len / x_input_len
        if (segment_middle_elements_len > 0):
            segment_middle_median = np.median(segment_middle_elements)
        else:
            return sys.maxsize    

        segments.append({'segment_elements': segment_middle_elements, 'segment_elements_len': segment_middle_elements_len, 
                        'segment_elements_ratio': segment_middle_elements_ratio, 'segment_median': segment_middle_median})  
        
    # Final segment
    segment_final_elements = y[(x_input >= x[(x_len - 1)])]
    segment_final_elements_len = len(segment_final_elements)
    segment_final_elements_ratio = segment_final_elements_len / x_input_len
    if (segment_final_elements_len > 0):
        segment_final_median = np.median(segment_final_elements)
    else:
        return sys.maxsize        

    segments.append({'segment_elements': segment_final_elements, 'segment_elements_len': segment_final_elements_len, 
                     'segment_elements_ratio': segment_final_elements_ratio, 'segment_median': segment_final_median})        

    # Comparing all segments with each other
    median_comparisons = []
    elems_ratio_comparisons = []

    segments_len = len(segments)

    for i in range(segments_len):
        median_comparison_i = 0.0
        elems_ratio_i_comparison = 0.0

        for j in range(i+1, segments_len):
            median_comparison_i = median_comparison_i + np.abs(segments[i]['segment_median'] - segments[j]['segment_median'])
            elems_ratio_i_comparison = elems_ratio_i_comparison + np.abs(segments[i]['segment_elements_ratio'] - segments[j]['segment_elements_ratio'])

        median_comparisons.append(median_comparison_i)
        elems_ratio_comparisons.append(elems_ratio_i_comparison)

    return (np.sum((median_comparisons - np.min(median_comparisons)) / (np.max(median_comparisons) - np.min(median_comparisons))) * -1.0 
                +
            np.sum(elems_ratio_comparisons) * unbalanced_cuts_importance)

def cut_variable_optimally_gen(x, y, unbalanced_cuts_importance = 1, seed = 16121993, n_parts = 2):
    min_x = np.min(x)
    max_x = np.max(x)

    unbalanced_cuts_importance = 1 if unbalanced_cuts_importance > 1 else np.abs(unbalanced_cuts_importance)     

    n_parts = np.round(n_parts)

    if n_parts == 0 or n_parts == 1:
        n_parts = 2
        warnings.warn('Warning: number of parts was equal to %d: changing number of parts to 2' % n_parts)     
    elif n_parts < 0:
        n_parts = np.abs(n_parts)
        warnings.warn('Warning: number of parts was negative: value converted to absolute') 

    if n_parts == 2:
        return cut_variable_optimally(x, y, maximize='median_diff')
    else:
        return differential_evolution(median_diff_gen, bounds = [(min_x, max_x)] * (n_parts - 1), args = [(x, y, unbalanced_cuts_importance)], maxiter = 1000, seed = seed)  
