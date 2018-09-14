import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
import warnings

'''
Performance report for binary categorical data (categories coded as 1 and 0)
given an array of real values and an array of predicted values.
'''
def predicted_report(y_test, y_pred):   
    results_to_vals = np.vectorize(lambda x: '1' if x == 1 else '0')

    y_test_str = results_to_vals(y_test)
    y_pred_str = results_to_vals(y_pred)
    
    print('%s\n' % pd.crosstab(y_test_str, y_pred_str, rownames=['Actual'], colnames=['Predicted'], margins=True))

    print(classification_report(y_test_str, y_pred_str))

'''
It plots percision-recall curve for binary categorical data (categories coded as 1 and 0) and returns
the best threshold for the chosen criterion. 

You can choose (optional) between the following criterions (if no criterion is chosen, 0.5 will be always returned):
- F1 score (criterion = 'F1')
- F2 score (criterion = 'F2')
- Lowest difference between precision and recall (metricriterion = 'min_prec-rec')
- Best F1 score with lowest difference between precision and recall (criterion = 'F1_min_p_r')
- Best F2 score with lowest difference between precision and recall (criterion = 'F2_min_p_r')
'''
def plot_precision_recall_curve(y_truth, y_pred, criterion=None):
    precision, recall, thresholds = precision_recall_curve(y_truth, y_pred)
  
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')

    plt.show() 

    if criterion == 'F1' or criterion == 'F1_min_p_r':
        f1_scores = 2 * ((precision * recall) / (precision + recall))
    elif criterion == 'F2' or criterion == 'F2_min_p_r':
        f2_scores = 5 * ((precision * recall) / ((4 * precision) + recall))

    if criterion is None:
        best_thresh = 0.5
        warnings.warn('Warning: returning 0.5 default threshold (no criterion chosen)')
    elif criterion == 'F1':            
        best_thresh = thresholds[np.argmax(f1_scores)]
        print('Threshold that gets best F1 score (%f): %f' % (np.max(f1_scores), best_thresh) )
    elif criterion == 'F2':
        best_thresh = thresholds[np.argmax(f2_scores)]
        print('Threshold that gets best F2 score (%f): %f' % (np.max(f2_scores), best_thresh) )
    elif criterion == 'min_prec-rec':
        diffs = np.abs(precision - recall)
        best_thresh = thresholds[np.argmin(diffs)]
        print('Threshold that gets lowest difference between precision and recall (%f): %f' % (np.min(diffs), best_thresh) )
    elif criterion == 'F1_min_p_r':
        f1_diffs = f1_scores - diffs
        best_thresh = thresholds[np.argmax(f1_diffs)]
        print('Threshold that gets lowest difference between precision and recall and best F1 (%f): %f' % (np.max(f1_diffs), best_thresh) )
    elif criterion == 'F2_min_p_r':
        f2_diffs = f2_scores - diffs
        best_thresh = thresholds[np.argmax(f2_diffs)]        
        print('Threshold that gets lowest difference between precision and recall and best F2 (%f): %f' % (np.max(f2_diffs), best_thresh) )
    
    return best_thresh

'''
Given a list of probabilities (usually from the positive class) and 
a threshold (0.5 by default), it returns the class of each probability
(1 if the i-th probability is greater than thresh, 0 otherwise)
'''
def cut_probs_with_thresh(probs, thresh = 0.5):
    cut_with_thresh = np.vectorize(lambda x: 1 if x >= thresh else 0)    
    return cut_with_thresh(probs)

'''
Full binary clasification report (it uses the best theshold according to the criterion specified)
'''
def full_binary_clasification_report(y_truth, y_pred, criterion=None):
    best_thresh = plot_precision_recall_curve(y_truth, y_pred, criterion)

    predicted_classes = cut_probs_with_thresh(y_pred)

    predicted_report(y_truth, predicted_classes)


'''
Auxiliar function to show the results of SkLearn Cross-Validation Grid-Search in a fancy way.

cv_results must be a result of cross_validate function.

scoring must be same scoring list used for the cross_validate function must be provided. 
The full possible list of scorings is the following: ['roc_auc', 'accuracy', 'f1', 'f2', 'precision', 'recall']

For each metric, the metric value for each fold and the mean (and its standard deviation) of the metric across all folds 
are printed.
'''
def k_folds_evaluation(cv_results, scorings):
    flag_no_valid_scorings = True

    if 'roc_auc' in scorings:
        flag_no_valid_scorings = False
        print('ROC AUC values: ')
        print(cv_results['test_roc_auc'])
        print('Mean ROC AUC: ')
        print('%0.3f (+/- %0.3f)' % (np.mean(cv_results['test_roc_auc']), np.std(cv_results['test_roc_auc'])))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    


    if 'accuracy' in scorings:
        flag_no_valid_scorings = False
        print('Accuracies: ')
        print(cv_results['test_accuracy'])
        print('Mean accuracy: ')
        print('%0.3f (+/- %0.3f)' % (np.mean(cv_results['test_accuracy']), np.std(cv_results['test_accuracy'])))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if 'f1' in scorings:
        flag_no_valid_scorings = False
        print('F1 values: ')
        print(cv_results['test_f1'])
        print('Mean F1: ')
        print('%0.3f (+/- %0.3f)' % (np.mean(cv_results['test_f1']), np.std(cv_results['test_f1'])))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if 'f2' in scorings:
        flag_no_valid_scorings = False
        print('F2 values: ')
        print(cv_results['test_f2'])
        print('Mean F1: ')
        print('%0.3f (+/- %0.3f)' % (np.mean(cv_results['test_f2']), np.std(cv_results['test_f2'])))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')    

    if 'precision' in scorings:
        flag_no_valid_scorings = False
        print('Precisions: ')
        print(cv_results['test_precision'])
        print('Mean precision: ')
        print('%0.3f (+/- %0.3f)' % (np.mean(cv_results['test_precision']), np.std(cv_results['test_precision'])))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if 'recall' in scorings:
        flag_no_valid_scorings = False
        print('Recalls: ')
        print(cv_results['test_recall'])
        print('Mean recall: ')
        print('%0.3f (+/- %0.3f)' % (np.mean(cv_results['test_recall']), np.std(cv_results['test_recall'])))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if flag_no_valid_scorings:
        warnings.warn('No valid scorings were provided')