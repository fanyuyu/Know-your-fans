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



def cluster_profile(df, cluster):
    '''
    Caculate the count, mean, std for each cluster as cluster profile

    Args:
        df: a data frame with all original features
        cluster: cluster lables from the clustering algorithm

    Returns:
        count, mean, std for each cluster
    ''' 
    return df[[cluster,'like_counts', 'comment_counts','share_counts', 'total_engagement']].groupby(cluster).agg(['count','mean', 'std'])




def bar_plot(profile):
    '''
    Draw a bar plot with mean and std for each cluster

    Args:
        profile: cluster profles with count, mean, std

    Returns:
        A bar plot 
    '''

    cluster_label = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5',
           'cluster 6', 'cluster 7', 'cluster 8', 'cluster 9', 'cluster 10']
    
    trace1 = go.Bar(
        x=cluster_label,
        y=profile['like_counts', 'mean'],
        name='like',
        error_y=dict(type='data', array=profile['like_counts', 'std'], visible=True
        )
    )

    trace2 = go.Bar(
        x=cluster_label,
        y=profile['comment_counts', 'mean'],
        name='comment',
        error_y=dict(type='data', array=profile['comment_counts', 'std'], visible=True
        )
    )

    trace3 = go.Bar(
        x=cluster_label,
        y=profile['share_counts', 'mean'],
        name='share',
        error_y=dict(type='data', array=profile['share_counts', 'std'], visible=True
        )
    )

    data = [trace1, trace2, trace3]
    layout = go.Layout(
        barmode='group'
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='error-bar-bar')
