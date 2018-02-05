# load all packages needed
import pandas as pd
import numpy as np
from scipy.linalg import solve
import networkx as nx
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



# Conduct cluster analysis on ONE page to identify active users
## Step 1: Subset data by page and week


def venn_counts(b):
    '''
    a function for set operation (linear equation) to get the unique users for each section in the venn diagram
    
    Arg:
        b: an array with all the counts for users who like, comment, and share, and the combinations of them
    
    Returns:
        x: an array with all the counts for users who only like, comment, or, share, 
           or who had two or more types of engagements  
    '''
    A = [[1, 0, 0, 1, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 1],
         [0, 0, 1, 0, 1, 1, 1],
         [0, 0, 0, 1, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1]]
    x = solve(A, b)
    return x[[0, 1, 3, 2, 4, 5, 6]].astype(int)


def df_subset(df1, df2, df3, page, week):
    '''
    subset data by page and week, and get basic information on user counts by engagement types
    
    Args:
        df1: the cleaned dataframe for LIKE
        df2: the cleaned dataframe for COMMENT
        df3: the cleaned dataframe for SHARE
        page: the target page_id
        week: the target week number
        
    Returns:
        like_p_w: dataframe 'like' by page and week
        comm_p_w: dataframe 'comment' by page and week
        shar_p_w: dataframe 'share' by page and week
    '''
    
    # subset data by page and week
    like_p_w = df1[(df1['page_id'] == page) & (df1['week'] == week)]
    comm_p_w = df2[(df2['page_id'] == page) & (df2['week'] == week)]
    shar_p_w = df3[(df3['page_id'] == page) & (df3['week'] == week)]
    
    
    # indentify users for each type of engagement and get counts
    like_user_id_uni = like_p_w['reaction_user_id'].unique()
    comm_user_id_uni = comm_p_w['comment_user_id'].unique()
    shar_user_id_uni = shar_p_w['shared_by_user_id'].unique()

    
    # identify users who engaged in two types and get counts
    like_comm_user_id_uni = np.intersect1d(like_user_id_uni, comm_user_id_uni)
    like_shar_user_id_uni = np.intersect1d(like_user_id_uni, shar_user_id_uni)
    comm_shar_user_id_uni = np.intersect1d(comm_user_id_uni, shar_user_id_uni)

    
    # identify users who engaged in three types and get counts
    like_comm_share_user_id_uni = reduce(np.intersect1d, (like_user_id_uni, comm_user_id_uni, shar_user_id_uni))

    
    # create a list to record the length of users who like, comment, share, or have more than two engagement
    counts = [len(like_user_id_uni), len(comm_user_id_uni), len(shar_user_id_uni),
              len(like_comm_user_id_uni), len(like_shar_user_id_uni), len(comm_shar_user_id_uni),
              len(like_comm_share_user_id_uni)]
    
    # draw a venn diagram to show the interationships between like, comment, and share 
    print '''The venn diagram shows the number of users who only like, comment, or share, \nand the number of users who had two or more types of engagements.'''
    v=venn3(subsets = counts, set_labels = ('Like', 'Comment', 'Share'))
    c=venn3_circles(subsets = counts, linestyle='dashed', linewidth=1, color="black")
    plt.show()
    print ''
    
    # conditional probabilities as a type of transition between engagements
    print 'The probability that a liker comments: ', round(float(counts[3])/counts[0], 3)
    print 'The probability that a commentor likes: ', round(float(counts[3])/counts[1], 3)
    print ''
    print 'The probability that a liker share: ', round(float(counts[4])/counts[0], 3)
    print 'The probability that a sharer likes: ', round(float(counts[4])/counts[2], 3)
    print ''
    print 'The probability that a commentor shares: ', round(float(counts[5])/counts[1], 3)
    print 'The probability that a sharer comments: ', round(float(counts[5])/counts[2], 3)
    
    return (like_p_w, comm_p_w, shar_p_w)


## Step 2: Extract features
def feature_extract(df1, df2, df3):
    '''
    Extract features based on each type of engagment and the interationships between them
    
    Args:
        df1: the cleaned and subsetted dataframe for LIKE
        df2: the cleaned and subsetted dataframe for COMMENT
        df3: the cleaned and subsetted dataframe for SHARE 
        
    Returns:
        A dataframe with all features
    '''
    
    # extract features of like
    like_features = df1.groupby('reaction_user_id').size().reset_index(name='like_counts')
    like_features = like_features.rename(columns={'reaction_user_id': 'user_id'})
    
    # extract features of comment
    comm_like_reply_count = df2[['comment_user_id','comment_like_count','comment_reply_count']].groupby('comment_user_id').agg('sum')
    comm_like_reply_count['comment_user_id'] = comm_like_reply_count.index
    comm_features = pd.merge(df2.groupby('comment_user_id').size().reset_index(name='comment_counts'),
                             comm_like_reply_count, on = 'comment_user_id')
    comm_features = comm_features.rename(columns={'comment_user_id': 'user_id'})
    
    # extract features of share
    shar_counts = df3.groupby('shared_by_user_id').size().reset_index(name='share_counts')
    shar_counts = shar_counts.rename(columns={'shared_by_user_id': 'user_id'})
    G_shar = nx.from_pandas_dataframe(df3, source = 'shared_by_user_id', target = 'shared_to_user_id', create_using = nx.DiGraph())
    
    out_degree = pd.DataFrame(nx.out_degree_centrality(G_shar).items(), columns=['user_id', 'out_degree'])
    closeness = pd.DataFrame(nx.closeness_centrality(G_shar).items(), columns=['user_id', 'closeness'])
    hub, authority = nx.hits(G_shar, max_iter=1000, tol=1e-05) # change the defaults to make the power iteration converge
    hub = pd.DataFrame(hub.items(), columns=['user_id', 'hub'])
    authority = pd.DataFrame(authority.items(), columns=['user_id', 'authority'])
    
    shar_features = reduce(lambda left,right: pd.merge(left, right, how = 'left', on='user_id'),
                           [shar_counts, out_degree, closeness, hub, authority])
    
    # join the above features
    like_comm_shar_features = reduce(lambda left,right: pd.merge(left, right, how = 'outer', on='user_id'),
                                     [like_features, comm_features, shar_features])
    
    # extract features for equal or more than two types of engagements
    like_comm_shar_features['like_and_comment'] = (pd.isnull(like_comm_shar_features['like_counts']) & 
                                                   pd.isnull(like_comm_shar_features['comment_counts'])).astype(int) 
    like_comm_shar_features['like_and_share'] = (pd.isnull(like_comm_shar_features['like_counts']) & 
                                                 pd.isnull(like_comm_shar_features['share_counts'])).astype(int) 
    like_comm_shar_features['share_and_comment'] = (pd.isnull(like_comm_shar_features['share_counts']) & 
                                                    pd.isnull(like_comm_shar_features['comment_counts'])).astype(int) 
    like_comm_shar_features['like_and_comment_and_share'] = (pd.isnull(like_comm_shar_features['like_counts']) & 
                                                             pd.isnull(like_comm_shar_features['comment_counts']) &
                                                             pd.isnull(like_comm_shar_features['share_counts'])).astype(int) 
    
    return like_comm_shar_features


## Step 3: Further subset data if only focusing on users who engaged >=2  a week in each type of engagement
def df_subset_twoORmore(df):
    '''
    Further subset data and only focus on users who engaged >=2 a week in each type of engagement
    
    Args:
        df: a dataframe with all features
    
    Returns:
        users who engaged >=2 a week in each type of engagement
    '''
    return(df.loc[(df_7976_51['like_counts'] > 1) |
                  (df_7976_51['comment_counts'] > 1) |
                  (df_7976_51['share_counts'] > 1), :])


## Step 4: Preprocess (Normalize) the features if needed
def feature_preprocss(df, norm = True):
    '''
    remove 'user_id' and normalize all features 
    
    Args:
        df: a dataframe with all features
        norm: whehter normalization is needed; the default is True
    
    Returns:
        a processed df for clustering
    '''
    
    # drop 'user_id' from the features
    df_features = df.drop('user_id', axis = 1).fillna(0)
    
    # standardize the features
    if norm:
        norm_features = StandardScaler().fit_transform(df_features) # normalized all features
        norm_features = pd.DataFrame(norm_features) # convert to a pd dataframe
        norm_features.columns = df_features.columns # add column names
        return(norm_features)
    else:
        return(df_features)