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
    # print('First segment:')
    # print('y[x_input < %f]' % x[0])
    segment_one_elements = y[x_input < x[0]]
    segment_one_elements_len = len(segment_one_elements)
    segment_one_elements_ratio = segment_one_elements_len / x_input_len
    if (segment_one_elements_len > 0):
        if metric == 0:
            segment_one_metric = np.median(segment_one_elements)
        elif metric == 1:
            segment_one_metric = np.mean(segment_one_elements)

        segments.append((segment_one_elements_ratio, segment_one_metric))    
    else:
        return sys.maxsize

    # print(segment_one_metric)

    # Middle segments
    for i in range(0, (x_len-1)):
        # print('Middle segment %d:' % i)
        # print('y[(x_input >= %f) & (x_input < %f)]' % (x[i], x[i+1]))
        segment_middle_elements = y[(x_input >= x[i]) & (x_input < x[i+1])]
        segment_middle_elements_len = len(segment_middle_elements)
        segment_middle_elements_ratio = segment_middle_elements_len / x_input_len
        if (segment_middle_elements_len > 0):
            if metric == 0:
                segment_middle_metric = np.median(segment_middle_elements)
            elif metric == 1:
                segment_middle_metric = np.mean(segment_middle_elements)            

            segments.append((segment_middle_elements_ratio, segment_middle_metric))                          
        else:
            return sys.maxsize    

        # print(segment_middle_metric)
        
    # Final segment
    # print('Final segment:')
    # print('y[x_input >= %f]' % x[(x_len - 1)])    
    segment_final_elements = y[(x_input >= x[(x_len - 1)])]
    segment_final_elements_len = len(segment_final_elements)
    segment_final_elements_ratio = segment_final_elements_len / x_input_len
    if (segment_final_elements_len > 0):
        if metric == 0:
            segment_final_metric = np.median(segment_final_elements)
        elif metric == 1:
            segment_final_metric = np.mean(segment_final_elements)         
  
        segments.append((segment_final_elements_ratio, segment_final_metric))       
    else:
        return sys.maxsize            

    # print(segment_final_metric)

    # Comparing all segments with each other
    segments_len = len(segments)
    n_comparisons = int(((segments_len * (segments_len - 1))) / 2)
    metric_comparisons = np.zeros(n_comparisons)
    elems_ratio_comparisons = np.zeros(n_comparisons)

    comparison_index = 0

    # print('*********')
    for i in range(segments_len):
        for j in range(i+1, segments_len):
            # print('%f - %f' % (segments[i][segment_metric], segments[j][segment_metric]))
            # print('%f' % abs(segments[i][segment_metric] - segments[j][segment_metric]))

            metric_comparisons[comparison_index] = abs(segments[i][segment_metric] - segments[j][segment_metric])
            # print(metric_comparisons[comparison_index])
            # print(metric_comparisons)

            elems_ratio_comparisons[comparison_index] = abs(segments[i][segment_elements_ratio] - segments[j][segment_elements_ratio])

            comparison_index = comparison_index+1
    # print(metric_comparisons)
    # print('*********')
    
    metric_comparisons_min = np.min(metric_comparisons)
    metric_comparisons_max = np.max(metric_comparisons)

    if (metric_comparisons_min == 0.0 and metric_comparisons_max == 0.0):
        return sum1d(elems_ratio_comparisons) * unbalanced_cuts_importance
    else:        
        metric_comparison_result_normalized = (metric_comparisons - metric_comparisons_min) /  (metric_comparisons_max - metric_comparisons_min)

        # print(metric_comparisons)
        # print(metric_comparison_result_normalized)
        # print(sum1d(metric_comparison_result_normalized) * -1.0)
        # print(sum1d(elems_ratio_comparisons) * unbalanced_cuts_importance)
        # print('...')
        # print(sum1d(metric_comparison_result_normalized) * -1.0 
        #             +
        #         sum1d(elems_ratio_comparisons) * unbalanced_cuts_importance)
        # print('------------------------------')

        return (sum1d(metric_comparison_result_normalized) * -1.0 
                    +
                sum1d(elems_ratio_comparisons) * unbalanced_cuts_importance)           

def metric_diff_gen(x, args):
    if len(args) > 1: 
        x_input, y, unbalanced_cuts_importance, metric = args[0], args[1], args[2], args[3]
    else:
        x_input, y, unbalanced_cuts_importance, metric = args[0][0], args[0][1], args[0][2], args[0][3]

    if metric == 'median_diff':
        metric = 0 
    elif metric == 'mean_diff':
        metric = 1

    return metric_diff_gen_numba(x, x_input, y, unbalanced_cuts_importance, metric)

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

# dev
# vars_utils.calculate_optimal_cuts(np.array([3, 4, 5, 6, 7, 8, 9, 10]), np.array([0, 0, 10, 10, 50, 50, 100, 100]), maximize='median_diff', n_parts = 4)
# a = np.random.rand(8)
# b = np.random.rand(8)
# vars_utils.calculate_optimal_cuts(a * 100, b * 100, maximize='median_diff', n_parts = 4)