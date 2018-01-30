# Know-your-fans

Pipeline to understand users' engagments on Facebook pages of large media company and to identify superfans. The company has more than 20 facebook pages, and these pages are mainly for shows or networks. There are three types of user engagements recorded: like a post, comment on a post, and share a post.  

The datasets include user IDs, page IDs, and post IDs, and their engagements on the posts. Specifically, data contains information on whether a user liked/commented/shared a post on a page, and if so, the timestamp of the engagment. 

First, EDA is performed to understand users' engagments on the page, post, and user levels. 
Insights from EDA:
  1. 
  2. 
  3. 
  
Second, data are subsetted based on page and time. 

Then, feature engineering is conducted to extract features in order to cluster users into different groups. The primary aim for clustering is to identify superfans who are actively engaged on the page.  

The main features considered are thirteen. They account both features for each type of engagment and features for the interrelationship between different types of engagements. For example, posts that are shared involved users "shared by" and "shared to". A network analysis is conducted to extract centrality measures, out-degree, closeness, authority, and hub, to measure users' performance in the sharing network. 


Main files:

main.py: the main function to call subroutines to run the pipeline in python2.7. The pipeline, including data subsetting, date merge, feature engineering, and clustering on three datesets, is automated for each page.  

feature_engineering.py:subroutine to conduct feature engineering. 
XXXXX
