# Know-your-fans

Pipeline to understand users' engagment on Facebook pages of large media company and to identify superfans. The company has more than 30 facebook pages, and these pages are mainly for shows or networks. There are three types of user engagements recorded: like a post, comment on a post, and share a post. The number of posts on each pages varies from 1 to more than 500.  
The datasets include user IDs, page IDs, and post IDs, and users' engagement in a month. Specifically, data contains information on whether a user liked/commented/shared a post on a page, and if so, the timestamp of the engagment. 

First, EDA is performed to understand users' engagment on the page, post, and user levels. 
Insights from EDA:
  1. Some pages are much more popular, which had more than 10,000+ users engaged in one month. 
  1. Majority of the users' engagement type was like, especailly for the most popular pages.
  2. There were a small proportion of users who had more than one type of engagements for each page. 
  3. There were a even smaller proportion of users who engaged across pages. 
  
Second, data were subsetted based on page and time. The identification of superfans were conducted on each page seperately becuse only a small proportion of users engaged acorss pages. Data were subsetted by time for the purpose of algorithm validation. 

Third, feature engineering is conducted to extract features in order to cluster users into different groups regarding their engagement pattern and active level. 
The main features considered are thirteen. They account both features for each type of engagment and features for the interrelationship between different types of engagements. For example, posts that are shared involved users "shared by" and "shared to". A network analysis was conducted to extract centrality measures, out-degree, closeness, authority, and hub, to measure users' performance in the sharing network. 

Fourth, two cluter algorithms, K-means and Gaussian mixture, were implemented. The primary aim for clustering is to identify superfans who are actively engaged on the page. 
PCA was used to reduce the dimension of the feature matrix and the clusters were plotted against the frist two dimensions of the projects matrix. 

At last, the seleted algorithm with the selected parameters were applied on the validation set, and results showed that the active clusters identified on the training set were relatively more active than other clusters in the following three weeks. 


Main files:

main.py: the main function to call subroutines to run the pipeline in python2.7. The pipeline, including data subsetting, date merge, feature engineering, and clustering on three datesets, is automated for each page.  

feature_engineering.py:subroutine to conduct feature engineering. 
XXXXX
