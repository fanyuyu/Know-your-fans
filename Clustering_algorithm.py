# load all packages needed
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn import mixture
from sklearn.preprocessing import StandardScaler, scale

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().magic(u'matplotlib inline')
from matplotlib_venn import venn3, venn3_circles
import seaborn as sns
sns.set()


def kmeans_elbow(df, max_cluster):
    '''
    implement the k-Means algorithm and output the elbow plot to decide the number of clusters
    
    Args:
        df: a dataframe with pre-processed features
        max_cluster: the maxi
        
    Returns:
        The elbow plot in terms of distortations
    
    '''
    distortions = []
    K = range(1,max_cluster+1)
    for i in K:
        kmeanModel = KMeans(n_clusters=i, init='k-means++').fit(df)
        kmeanModel.fit(df)
        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Distortation of kmeans across number of clusters')
    plt.show()


def kmeans_k_cluster(df, k):
    '''
    implement the k-Means algorithm with the number of clusters chosen
    
    Args:
        df: a dataframe with pre-processed features
        k: the number of clusters
        
    Returns:
        The cluster labels
    
    '''
    
    # fit the model with k clusters
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(df)
    
    # predict the cluster labels 
    cluster = kmeans.predict(df)
    
    return(cluster)



def gmm_elbow(df, max_cluster):
    '''
    implement the Gaussian mixture algorithm and output the elbow plot to decide the number of clusters
    
    Args:
        df: a dataframe with pre-processed features
        max_cluster: the maxi
        
    Returns:
        The elbow plot in terms of AIC and BIC
    
    '''
    
    n_components = np.arange(1, max_cluster+1)
    models = [mixture.GaussianMixture(n, covariance_type='full').fit(df) for n in n_components]

    plt.plot(n_components, [m.bic(df) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(df) for m in models], label='AIC')
    plt.title('AIC and BIC of Gaussian mixture model across number of clusters')
    plt.legend(loc='best')
    plt.xlabel('n_components')


def gmm_k_cluster(df, k):
    '''
    implement the Gaussian mixture algorithm with the number of clusters chosen
    
    Args:
        df: a dataframe with pre-processed features
        k: the number of clusters
        
    Returns:
        The cluster labels
    
    '''
    
    # fit the model with k clusters
    model = GaussianMixture(n_components=k).fit(df)
    
    # predict the cluster labels and add into the df
    cluster = model.predict(df)
    
    return(cluster)
