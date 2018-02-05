# load all packages needed
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set() 
from scipy import linalg
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


def pca_k_component(df, k, cluster_label):
    '''
    apply pac for dimension reduction to visualize the clusters
    
    Args:
        df: a dataframe with all features
        k: the number of component
        cluster_label: the cluster labels from kmeans or gmm
    
    Return:
        All necessary information from PCA and a 2D plot based on the projection matrix
    
    '''
    pca = PCA(n_components=k).fit(xx_prepro)
    print 'The amount of variance explained by each of the selected components: ', pca.explained_variance_.round(2)
    print 'The Percentage of variance explained by each of the selected comonents: ', pca.explained_variance_ratio_.round(2)
    print 'Coefficients for the first two components: '
    print pca.components_.round(2)
    print ''
    projected = PCA(2).fit_transform(xx_prepro)
    
    plt.scatter(projected[:, 0], projected[:, 1],
            c=cluster, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 3))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')        
    plt.colorbar()
    
    
