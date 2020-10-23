import time
%matplotlib inline 
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (15, 9)
import seaborn as sns
import pandas as pd
import numpy as np

import pandas as pd
start = 0
interval = 10
scores_df = pd.read_csv("./objects/scores_df.csv")
mean_scores = pd.DataFrame(columns =['score'])
actions_df = pd.read_csv("./objects/actions_df.csv")
max_scores = pd.DataFrame(columns =['max_score'])
q_max_scores = pd.DataFrame(columns =['q_max'])
while interval <= len(scores_df):
    mean_scores.loc[len(mean_scores)] = (scores_df.loc[start:interval].mean()['scores'])
    max_scores.loc[len(max_scores)] = (scores_df.loc[start:interval].max()['scores'])
    start = interval
    interval = interval + 10

q_max_df = pd.read_csv("./objects/q_values.csv")

start = 0
interval = 1000
while interval <=len(q_max_df):
    q_max_scores.loc[len(q_max_scores)] = (q_max_df.loc[start:interval].mean()['actions'])
    start = interval
    interval = interval + 1000
    
mean_scores.plot()
max_scores.plot()
q_max_scores.plot()