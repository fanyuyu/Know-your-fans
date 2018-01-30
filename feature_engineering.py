
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


########## step1: subset data by page and week ########## 
# define a function to caculate the counts for each section in the venn diagram
def venn_counts(b):
    A = [[1, 0, 0, 1, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 1],
         [0, 0, 1, 0, 1, 1, 1],
         [0, 0, 0, 1, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1]]
    x = solve(A, b)
    return x[[0, 1, 3, 2, 4, 5, 6]].astype(int)


# subset data by page and week, and get basic information on user counts by engagement types
def df_subset(df1, df2, df3, page, week):
	# df1: the cleaned dataframe for LIKE
    # df2: the cleaned dataframe for COMMENT
    # df3: the cleaned dataframe for SHARE 
    # page: the target page_id 
    # week: the target week number

    # subset data by page and week
    like_p_w = df1[(df1['page_id'] == page) & (df1['week'] == week)]
    comm_p_w = df2[(df2['page_id'] == page) & (df2['week'] == week)]
    shar_p_w = df3[(df3['page_id'] == page) & (df3['week'] == week)]
    
    # indentify users for each type of engagement and get counts
    like_user_id_uni = like_p_w['reaction_user_id'].unique()
    comm_user_id_uni = comm_p_w['comment_user_id'].unique()
    shar_user_id_uni = shar_p_w['shared_by_user_id'].unique()
    len1 = len(like_user_id_uni)
    len2 = len(comm_user_id_uni)
    len3 = len(shar_user_id_uni)
    # print 'number of users who like: ', len1  
    # print 'number of users who comment: ', len2
    # print 'number of users who share: ', len3
    # print ''
    
    # identify users who engaged in two types and get counts
    like_comm_user_id_uni = np.intersect1d(like_user_id_uni, comm_user_id_uni)
    like_shar_user_id_uni = np.intersect1d(like_user_id_uni, shar_user_id_uni)
    comm_shar_user_id_uni = np.intersect1d(comm_user_id_uni, shar_user_id_uni)
    len4 = len(like_comm_user_id_uni)
    len5 = len(like_shar_user_id_uni)
    len6 = len(comm_shar_user_id_uni)
    # print 'number of users who like and comment: ', len4
    # print 'number of users who like and share: ', len5
    # print 'number of users who comment and share: ', len6
    # print ''
    
    # identify users who engaged in three types and get counts
    like_comm_share_user_id_uni = reduce(np.intersect1d, (like_user_id_uni, comm_user_id_uni, shar_user_id_uni))
    len7 = len(like_comm_share_user_id_uni)
    # print 'number of users who like, comment, and share: ', len7
    # print ''
    
    # draw a venn diagram to show the interationships between like, comment, and share 
    print '''The venn diagram shows the number of users who only like, comment, or share, \nand the number of users who had two or more types of engagements.'''
    set_counts = venn_counts([len1, len2, len3, len4, len5, len6, len7])
    v=venn3(subsets = set_counts, set_labels = ('Like', 'Comment', 'Share'))
    c=venn3_circles(subsets = set_counts, linestyle='dashed', linewidth=1, color="black")
    plt.show()
    print ''
    
    # conditional probabilities as a type of transition between engagements
    conP_comm_like = round(float(len4)/len1, 3)
    comP_like_comm = round(float(len4)/len2, 3)
    
    conP_shar_like = round(float(len5)/len1, 3)
    comP_like_shar = round(float(len5)/len3, 3)
    
    conP_shar_comm = round(float(len6)/len2, 3)
    comP_comm_shar = round(float(len6)/len3, 3)
    
    print 'The probability that a liker comments: ', conP_comm_like
    print 'The probability that a commentor likes: ', comP_like_comm
    print ''
    print 'The probability that a liker share: ', conP_shar_like
    print 'The probability that a sharer likes: ', comP_like_shar
    print ''
    print 'The probability that a commentor shares: ', conP_shar_comm
    print 'The probability that a sharer comments: ', comP_comm_shar
    
    return (like_p_w, comm_p_w, shar_p_w)


########## step 2: extract features form like, comment, and share, and from their inter-relationships ########## 
def feature_extract(df1, df2, df3):
    # df1: the cleaned and subsetted dataframe for LIKE
    # df2: the cleaned and subsetted dataframe for COMMENT
    # df3: the cleaned and subsetted dataframe for SHARE 
    
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



