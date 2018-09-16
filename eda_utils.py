from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from minepy import MINE
import pandas as pd

'''
Correlation plot given a dataframe. Only the upper triangle is shown.

You can choose the plot size (plot_size) and the correlation method/coefficient to use:

Linear correlation:
- pearson (by default)
- kendall
- spearman

Other kinds of correlation:
- mic (https://livebook.datascienceheroes.com/exploratory_data_analysis/correlation.html#adding-noise)
- mic-r2 (https://livebook.datascienceheroes.com/exploratory_data_analysis/correlation.html#measuring-non-linearity-mic-r2)

(https://en.wikipedia.org/wiki/Maximal_information_coefficient)
(https://www.ncbi.nlm.nih.gov/pubmed/22174245)
(https://cran.r-project.org/web/packages/minerva/minerva.pdf)
'''
def correlation_plot(df, plot_size = (20, 20), method = 'pearson'):
    if method in ['pearson', 'kendall', 'spearman']:
        corr = df.corr(method = method)
    elif method == 'mic':
        corr = mic(df)
    elif method == 'mic-r2':
        corr = mic_r2(df)

    df_cols = list(df.columns)    

    plt.figure(figsize=plot_size)

    drop_self = np.zeros_like(corr)
    drop_self[np.tril_indices_from(drop_self, k = -1)] = True

    ax = sns.heatmap(corr, xticklabels = df_cols, yticklabels = df_cols, 
                annot = True, fmt = '.2f', cmap = 'RdBu_r', mask = drop_self
    )

    ax.set_title(method, fontsize = 20)

    plt.show()

    return corr    

def mic(df):
    mine = MINE(alpha = 0.6, c = 15, est = 'mic_approx')
    
    mic_dict = {}
    
    for col_a in list(df.columns):
        mic_dict[col_a] = []
        
        for col_b in list(df.columns):
            mine.compute_score(df[col_a].values, df[col_b].values)
            mic_dict[col_a].append( mine.mic() )
    
    return pd.DataFrame(data = mic_dict, index = df.columns)

def mic_r2(df):    
    mine = MINE(alpha = 0.6, c = 15, est = 'mic_approx')
    
    mic_dict = {}
    
    for col_a in list(df.columns):
        mic_dict[col_a] = []
        
        for col_b in list(df.columns):
            mine.compute_score(df[col_a].values, df[col_b].values)
            mic_dict[col_a].append( mine.mic() )
    
    return pd.DataFrame(data = mic_dict, index = df.columns) - np.abs(df.corr())
