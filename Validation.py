# load all packages needed
import pandas as pd
import numpy as np

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().magic(u'matplotlib inline')
from matplotlib_venn import venn3, venn3_circles
import seaborn as sns
sns.set() 
from scipy import linalg
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls



## Validate the algorithm by checking the clusters activities in the following weeks
cluster1_user_ID = df['user_id'][cluster == 0]
cluster2_user_ID = df['user_id'][cluster == 1]
cluster3_user_ID = df['user_id'][cluster == 2]


like_p_w, comm_p_w, shar_p_w = df_subset(like, comm, shar, 'p1', 52)
df_p1_52 = feature_extract(like_p_w, comm_p_w, shar_p_w)


like_p_w, comm_p_w, shar_p_w = df_subset(like, comm, shar, 'p1', 1)
df_p1_1 = feature_extract(like_p_w, comm_p_w, shar_p_w)


like_p_w, comm_p_w, shar_p_w = df_subset(like, comm, shar, 'p1', 2)
df_p1_2 = feature_extract(like_p_w, comm_p_w, shar_p_w)



def engage_validt(df_new, df_old, cluster_label):
    '''
    Validate the engagment level for each cluster
    
    Args:
        df_new: the new dataframe for validation
        df_old: the old dataframe for training
        cluster lable: the label obtained from the training
        
    Returns:
        engagement levels for different clusters on the new dataframe
    
    '''
    
    # identify user_ids for each cluster on the train
    cluster1_user = df_old['user_id'][cluster_label == 0]
    cluster2_user = df_old['user_id'][cluster_label == 1]
    cluster3_user = df_old['user_id'][cluster_label == 2]
    
    # identify user_ids for each cluster on the validation
    cluster1_user_new = df_new['user_id'].isin(cluster1_user)
    cluster2_user_new = df_new['user_id'].isin(cluster2_user)
    cluster3_user_new = df_new['user_id'].isin(cluster3_user)
    
    # output descriptive statistics on the validiation regarding their engagments
    engage_new = pd.DataFrame(df_new[cluster1_user_new][['like_counts', 'comment_counts', 'share_counts']].sum(axis=1).describe(),
                             columns=['cluster1'])
    
    engage_new['cluster2'] = df_new[cluster2_user_new][['like_counts', 'comment_counts', 'share_counts']].sum(axis=1).describe()
    engage_new['cluster3'] = df_new[cluster3_user_new][['like_counts', 'comment_counts', 'share_counts']].sum(axis=1).describe()
                              
    return(engage_new.round(2))




# Calculate the total engagement for week52, week1, and week2
week52_engage = engage_validt(df_p1_52, df, cluster)
week1_engage = engage_validt(df_p1_1, df, cluster)
week2_engage = engage_validt(df_p1_2, df, cluster)

