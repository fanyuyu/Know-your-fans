
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


# load data and conduct exploratory analyses
like = pd.read_csv('reactions_1m.csv')
comm = pd.read_csv('comments_1m.csv')
shar = pd.read_csv('sharetags_1m.csv')


def change_dtype(df):
    '''
    change dtypes for id and time
    
    Args:
        df: a dataframe
    
    Returns:
        the dataframe with correct dtypes
    '''
    for i in df.columns:
        if 'id' in i:
            df[i] = df[i].astype(str)
        if 'time' in i:
            df[i] = pd.to_datetime(df[i])
    return df


## Examine "like_reactions"
# change the dtypes
like = change_dtype(like)

# the basic information of like_reactions
print like.info()
print like.head()


### Information about reactions
print 'Type of reactions users have: ', like.reaction_type.unique()
print 'Probs of each reaction type: ', (like['reaction_type'].value_counts().values / float(len(like))).round(2)


# check the distribution of like reaction regardless of page
data = [go.Bar(
            x = like['reaction_type'].value_counts().index.values,
            y = like['reaction_type'].value_counts().values / float(len(like)),
            text='Reaction'
    )]

layout = go.Layout(
    title = 'Distribution of like reaction'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='liketion_bar')


### Information about page

page_title = like['page_id'].value_counts().index.values
print page_title
page_prob = (like['page_id'].value_counts().values / float(len(like))).round(2)
print page_prob


# Check the distribution of page
data = [go.Bar(
            x = page_title,
            y = page_prob,
            text='Viacome Facebook page'
    )]

layout = go.Layout(
    title = 'Distribution of Likes on Viacom Facebook Page',
    xaxis=dict(
        type = 'category'  # this line is very critical to make page_id as string!    
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='page_id_bar')


### Distribution of like reactions by page
page_probs = like.groupby('page_id').size().div(len(like))
reac_page_probs = like.groupby(['page_id', 'reaction_type']).size().div(len(like)).div(page_probs, axis=0, level='page_id')
reac_page_probs_df = pd.DataFrame(reac_page_probs) # convert pandas multi-column series to a data frame
reac_page_probs_df.reset_index(inplace=True) # reset_index to make index aligned
reac_page_probs_df = reac_page_probs_df.rename(columns = {0:'prob'}) # change the column name from 0 to 'prob'


trace1 = go.Bar(
            x = reac_page_probs_df[reac_page_probs_df.reaction_type=='LIKE']['page_id'],
            y = reac_page_probs_df[reac_page_probs_df.reaction_type=='LIKE']['prob'],
            name = 'LIKE'
    )

trace2 = go.Bar(
            x = reac_page_probs_df[reac_page_probs_df.reaction_type=='HAHA']['page_id'],
            y = reac_page_probs_df[reac_page_probs_df.reaction_type=='HAHA']['prob'],
            name = 'HAHA'
    )

trace3 = go.Bar(
            x = reac_page_probs_df[reac_page_probs_df.reaction_type=='LOVE']['page_id'],
            y = reac_page_probs_df[reac_page_probs_df.reaction_type=='LOVE']['prob'],
            name = 'LOVE'
    )

trace4 = go.Bar(
            x = reac_page_probs_df[reac_page_probs_df.reaction_type=='WOW']['page_id'],
            y = reac_page_probs_df[reac_page_probs_df.reaction_type=='WOW']['prob'],
            name = 'WOW'
    )

trace5 = go.Bar(
            x = reac_page_probs_df[reac_page_probs_df.reaction_type=='ANGRY']['page_id'],
            y = reac_page_probs_df[reac_page_probs_df.reaction_type=='ANGRY']['prob'],
            name = 'ANGRY'
    )

trace6 = go.Bar(
            x = reac_page_probs_df[reac_page_probs_df.reaction_type=='SAD']['page_id'],
            y = reac_page_probs_df[reac_page_probs_df.reaction_type=='SAD']['prob'],
            name = 'SAD'
    )

data = [trace1, trace2, trace3, trace4, trace5, trace6]

layout = go.Layout(
    title = 'Distribution of like reactions by Viacom Facebook Page',
    xaxis=dict(
        type = 'category'
    ),
    barmode = 'stack',
    # showlegend = False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='page_id_like_bar')



## Examine "comment"

# change the dtypes
comm = change_dtype(comm)

# the basic information of comment
print comm.info()
print comm.head()


### Distriution of comment counts by page
data = [go.Bar(
            x = comm['page_id'].value_counts().index.values,
            y = comm['page_id'].value_counts().values / float(len(comm['page_id'])),
            text='Viacome Facebook page'
    )]

layout = go.Layout(
    title = 'Distribution of Comment Counts on Viacom Facebook Page',
    xaxis=dict(
        type = 'category'  # this line is very critical to make page_id as string!
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='comm_page_bar')


## Examine "share"

# change the dtypes
shar = change_dtype(shar)

# the basic information of comment
print shar.info()
print shar.head()


### Distribution of share counts by page
data = [go.Bar(
            x = shar['page_id'].value_counts().index.values,
            y = shar['page_id'].value_counts().values / float(len(shar['page_id'])),
            text='Viacome Facebook page'
    )]

layout = go.Layout(
    title = 'Distribution of Share Counts on Viacom Facebook Page',
    xaxis=dict(
        type = 'category'  # this line is very critical to make page_id as string!
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='shar_page_bar')


## Aggregate the three types of engagements on page level to examine post information
like_post_byPage = like[['page_id','post_id']].groupby(['page_id']).agg(['nunique']).unstack()
like_post_byPage = like_post_byPage.reset_index().rename(columns={0: 'like_post_counts'})

comm_post_byPage = comm[['post_id','page_id']].groupby(['page_id']).agg(['nunique']).unstack()
comm_post_byPage = comm_post_byPage.reset_index().rename(columns={0: 'comm_post_counts'})

shar_post_byPage = shar[['post_id','page_id']].groupby(['page_id']).agg(['nunique']).unstack()
shar_post_byPage = shar_post_byPage.reset_index().rename(columns={0: 'shar_post_counts'})

like_comm_shar_byPage = pd.merge(pd.merge(like_post_byPage[['page_id', 'like_post_counts']],
                                          comm_post_byPage[['page_id', 'comm_post_counts']],
                                          how='outer',
                                          on='page_id'),
                                 shar_post_byPage[['page_id', 'shar_post_counts']],
                                          how='outer',
                                          on='page_id')
like_comm_shar_byPage['mean_counts'] = like_comm_shar_byPage.loc[:,'like_post_counts':'shar_post_counts'].mean(axis=1)


# In[31]:


like_comm_shar_byPage.sort_values('mean_counts', ascending=False)


## Examine the relationships between different engagements

### The entire sample
like_user_uni_ID = like['reaction_user_id'].unique()
comm_user_uni_ID = comm['comment_user_id'].unique()
shar_user_uni_ID = shar['shared_by_user_id'].unique()
print 'number of users who like: ', len(like_user_uni_ID)
print 'number of users who comment: ', len(comm_user_uni_ID)
print 'number of users who share: ', len(shar_user_uni_ID)
print ''

like_comm_user_uni_ID = np.intersect1d(like_user_uni_ID, comm_user_uni_ID)
like_shar_user_uni_ID = np.intersect1d(like_user_uni_ID, shar_user_uni_ID)
comm_shar_user_uni_ID = np.intersect1d(comm_user_uni_ID, shar_user_uni_ID)
print 'number of users who like and comment: ', len(like_comm_user_uni_ID)
print 'number of users who like and share: ', len(like_shar_user_uni_ID)
print 'number of users who comment and share: ', len(comm_shar_user_uni_ID)
print ''

like_comm_shar_uni_ID = reduce(np.intersect1d, (like_user_uni_ID, comm_user_uni_ID, shar_user_uni_ID))
print 'number of users who like, comment, and share: ', len(like_comm_shar_uni_ID)



### Each page
# ** The Results for each page will be shown when fitting functions for each page

## Join the three tables and subset data for better visulization using SuperSet
# ** The tables were joined using prestoDB on AWS athena

### Load the data and perform basic preprocessing
like_comm_shar = pd.read_csv('like_comm_shar_all.csv')
like_comm_shar = change_dtype(like_comm_shar)
like_comm_shar = like_comm_shar.replace('nan', np.NaN)


like_comm_shar.info()


### Create three indicator variables for the three types of engagements
like_comm_shar['comment'] = pd.notnull(like_comm_shar['comment_created_time']).astype(int)
like_comm_shar['share'] = pd.notnull(like_comm_shar['shared_created_time']).astype(int)
like_comm_shar['like'] = pd.notnull(like_comm_shar['reaction_type']).astype(int)

like_comm_shar.to_csv('like_comm_shar_all_py.csv', index=False)


### subset data by pages (6 popular pages to focus on) 
like_comm_shar_all_6pages = like_comm_shar[(like_comm_shar.page_id == 'p1') |
                                           (like_comm_shar.page_id == 'p2') |
                                           (like_comm_shar.page_id == 'p3') |
                                           (like_comm_shar.page_id == 'p4') |
                                           (like_comm_shar.page_id == 'p5') |
                                           (like_comm_shar.page_id == 'p6')]

print len(set(like_comm_shar_all_6pages.user_id))
print like_comm_shar_all_6pages.shape

like_comm_shar_all_6pages.to_csv('like_comm_shar_all_6pages_py.csv', index=False)


### subset data to select pages corresponding to 4 shows
like_comm_shar_all_4shows = like_comm_shar[(like_comm_shar.page_id == 'p1') |
                                           (like_comm_shar.page_id == 'p2') |
                                           (like_comm_shar.page_id == 'p3') |
                                           (like_comm_shar.page_id == 'p4')]

print len(set(like_comm_shar_all_4shows.user_id))
print like_comm_shar_all_4shows.shape


like_comm_shar_all_4shows.to_csv('like_comm_shar_all_4shows.csv', index=False)


### subset data to select pages corresponding to 2 networks
like_comm_shar_all_2networks = like_comm_shar[(like_comm_shar.page_id == 'p5') |
                                              (like_comm_shar.page_id == 'p6')]

print len(set(like_comm_shar_all_2networks.user_id))
print like_comm_shar_all_2networks.shape


like_comm_shar_all_2networks.to_csv('like_comm_shar_all_2networks.csv', index=False)