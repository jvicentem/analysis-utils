import numba
from numba import f8
import numpy as np
from scipy.optimize import minimize_scalar  
from scipy.optimize import differential_evolution
import sys
import warnings       

@numba.jit('f8(f8[:])')
def sum1d(array):
    sum = 0.0
    for i in range(array.shape[0]):
        sum += array[i]
    return sum

@numba.jit(nopython=True)
def metric_diff_gen_numba(x, x_input, y, unbalanced_cuts_importance, metric):
    segment_elements_ratio = 0
    segment_metric = 1

    x_len = len(x)

    segments = []

    x_input_len = len(x_input)

    # First segment
    segment_one_elements = y[x_input < x[0]]
    segment_one_elements_len = len(segment_one_elements)
    segment_one_elements_ratio = segment_one_elements_len / x_input_len
    if (segment_one_elements_len > 0):
        if metric == 0:
            segment_one_metric = np.median(segment_one_elements)
        elif metric == 1:
            segment_one_metric = np.mean(segment_one_elements)

        segments.append([segment_one_elements_ratio, segment_one_metric])    
    else:
        return sys.maxsize

    # Middle segments
    for i in range(0, (x_len-1)):
        segment_middle_elements = y[(x_input >= x[i]) & (x_input < x[i+1])]
        segment_middle_elements_len = len(segment_middle_elements)
        segment_middle_elements_ratio = segment_middle_elements_len / x_input_len
        if (segment_middle_elements_len > 0):
            if metric == 0:
                segment_middle_metric = np.median(segment_middle_elements)
            elif metric == 1:
                segment_middle_metric = np.mean(segment_middle_elements)            

            segments.append([segment_middle_elements_ratio, segment_middle_metric])                          
        else:
            return sys.maxsize    
        
    # Final segment
    segment_final_elements = y[(x_input >= x[(x_len - 1)])]
    segment_final_elements_len = len(segment_final_elements)
    segment_final_elements_ratio = segment_final_elements_len / x_input_len
    if (segment_final_elements_len > 0):
        if metric == 0:
            segment_final_metric = np.median(segment_middle_elements)
        elif metric == 1:
            segment_final_metric = np.mean(segment_middle_elements)         
  
        segments.append([segment_final_elements_ratio, segment_final_metric])       
    else:
        return sys.maxsize            

    # Comparing all segments with each other
    metric_comparisons = []
    elems_ratio_comparisons = []

    segments_len = len(segments)

    for i in range(segments_len):
        for j in range(i+1, segments_len):
            metric_comparisons.append(abs(segments[i][segment_metric] - segments[j][segment_metric]))
            elems_ratio_comparisons.append(abs(segments[i][segment_elements_ratio] - segments[j][segment_elements_ratio]))

    metric_comparison_result_normalized = -1 
    # metric_comparison_result_normalized = (np.sum((metric_comparisons - np.min(metric_comparisons)) / (np.max(metric_comparisons) - np.min(metric_comparisons))))        
    metric_comparison_result_normalized = 0 if np.isnan(metric_comparison_result_normalized) else metric_comparison_result_normalized

    elems_ratio_comparisons = np.asarray(elems_ratio_comparisons)

    print(elems_ratio_comparisons)

    return (metric_comparison_result_normalized * -1.0 
                +
            sum1d(elems_ratio_comparisons) * unbalanced_cuts_importance)    

def metric_diff_gen(x, args):
    # segment_elements_ratio = 0
    # segment_metric = 1

    # x_len = len(x)

    # segments = []

    if len(args) > 1: 
        x_input, y, unbalanced_cuts_importance, metric = args[0], args[1], args[2], args[3]
    else:
        x_input, y, unbalanced_cuts_importance, metric = args[0][0], args[0][1], args[0][2], args[0][3]

    if metric == 'median_diff':
        metric = 0 
    elif metric == 'mean_diff':
        metric = 1

    return metric_diff_gen_numba(x, x_input, y, unbalanced_cuts_importance, metric)

    # x_input_len = len(x_input)

    # # First segment
    # segment_one_elements = y[x_input < x[0]]
    # segment_one_elements_len = len(segment_one_elements)
    # segment_one_elements_ratio = segment_one_elements_len / x_input_len
    # if (segment_one_elements_len > 0):
    #     segment_one_metric = fun(segment_one_elements)

    #     segments.append([segment_one_elements_ratio, segment_one_metric])    
    # else:
    #     return sys.maxsize

    # # Middle segments
    # for i in range(0, (x_len-1)):
    #     segment_middle_elements = y[(x_input >= x[i]) & (x_input < x[i+1])]
    #     segment_middle_elements_len = len(segment_middle_elements)
    #     segment_middle_elements_ratio = segment_middle_elements_len / x_input_len
    #     if (segment_middle_elements_len > 0):
    #         segment_middle_metric = fun(segment_middle_elements)

    #         segments.append([segment_middle_elements_ratio, segment_middle_metric])                          
    #     else:
    #         return sys.maxsize    
        
    # # Final segment
    # segment_final_elements = y[(x_input >= x[(x_len - 1)])]
    # segment_final_elements_len = len(segment_final_elements)
    # segment_final_elements_ratio = segment_final_elements_len / x_input_len
    # if (segment_final_elements_len > 0):
    #     segment_final_metric = fun(segment_final_elements)
  
    #     segments.append([segment_final_elements_ratio, segment_final_metric])       
    # else:
    #     return sys.maxsize            

    # # Comparing all segments with each other
    # metric_comparisons = []
    # elems_ratio_comparisons = []

    # segments_len = len(segments)

    # for i in range(segments_len):
    #     for j in range(i+1, segments_len):
    #         metric_comparisons.append(np.abs(segments[i][segment_metric] - segments[j][segment_metric]))
    #         elems_ratio_comparisons.append(np.abs(segments[i][segment_elements_ratio] - segments[j][segment_elements_ratio]))

    # metric_comparison_result_normalized = (np.sum((metric_comparisons - np.min(metric_comparisons)) / (np.max(metric_comparisons) - np.min(metric_comparisons))))
    # metric_comparison_result_normalized = 0 if np.isnan(metric_comparison_result_normalized) else metric_comparison_result_normalized

    # return (metric_comparison_result_normalized * -1.0 
    #             +
    #         np.sum(elems_ratio_comparisons) * unbalanced_cuts_importance)

'''
Given two numeric variables (one of them is the target variable),
it discretizes the non target variable so it maximizes the mean or median
difference between categories 
'''
def calculate_optimal_cuts(x, y, unbalanced_cuts_importance = 1, maximize='mean_diff', seed = 16121993, n_parts = 2, max_iter = 1000):
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

    if maximize == 'median_diff' or maximize == 'mean_diff':
        return differential_evolution(metric_diff_gen, bounds = [(min_x, max_x)] * (n_parts - 1), 
                                        args = [(x, y, unbalanced_cuts_importance, maximize)], 
                                        maxiter = max_iter, seed = seed).x  
    else:
        warnings.warn('Invalid maximization method.') 
# vars_utils.calculate_optimal_cuts(np.array([3, 4, 5, 6, 7, 8, 9, 10]), np.array([0, 0, 10, 10, 50, 50, 100, 100]), maximize='median_diff', n_parts = 4)