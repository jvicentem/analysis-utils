from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from minepy import MINE
import pandas as pd

def linear_correlation_plot(df, plot_size = (20, 20)):
    corr = df.corr()

    df_cols = list(df.columns)

    plt.figure(figsize=plot_size)
    sns.heatmap(np.triu(corr, k=1), xticklabels=df_cols, yticklabels=df_cols, annot=True, fmt='.2f')

    return corr    

def calculate_mic(df):
    mine = MINE(alpha = 0.6, c = 15, est = 'mic_approx')
    
    mic_dict = {}
    
    cols = list(df.columns)
    
    for col_a in list(df.columns):
        mic_dict[col_a] = []
        
        for col_b in cols:
            mine.compute_score(df[col_a].values, df[col_b].values)
            mic_dict[col_a].append( mine.mic() )

    cols.remove(col_a)
    
    return pd.DataFrame(data = mic_dict, index = df.columns)
