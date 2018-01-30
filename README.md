# Know-your-fans

Pipeline to understand users' engagments on Facebook pages of large media company and to identify superfans. The company has more than 20 facebook pages, and these pages are mainly for shows or networks. There are three types of user engagements recorded: like a post, comment on a post, and share a post.  

The datasets include user IDs, page IDs, and post IDs, and their engagements on the posts. Specifically, data contains information on whether a user liked/commented/shared a post on a page, and if so, the timestamp of the engagment. 

First, EDA is performed to understand users' engagments on the page, post, and user levels. 
Insights from EDA:
  1. 
  2. 
  3. 
  
Then, data are subsetted based on page and time. After, feature engineering is conducted to extract features in order to cluster users into different groups regarding their active level. 

The main features considered are thirteen. They account both features on each engagment type for the number of people connected to the first two pools used by each user and for the time elapsed between the first two payments and the first two invitations (sent or received) by each user. The idea behind the computation of these features is that:

Engaged customers contribute to pools that are linked to more people respect to unengaged users (network effect). See Fig3.png, Fig4.png
Engaged customers use the product more frequently than unengaged users. This different behavior is already present from their first uses of the product. See Fig7.png
Main files:

main.py: Main, interactive file to run the pipeline in python3. It calls subroutines in two files (alldata.py and random_forest.py). These files are used to load, merge and analyze four datasets, and to compute predictions using a Random Forest Classifier (using scikit-learn).

alldata.py: This file contains three subroutines (merge_data, add_features, make_labels).

merge_data: subroutine to merge data for user IDs, invitations and payments
add_features: subroutine to compute new features from the early activity of each user
make_labels: subroutine to label all data and output final dataset to be used for Machine Learning predictions
ml_algo.py: This file computes predictions using a Random Forest Classifier, together with multiple summary statisitics such as accuracy, precision, recall, F1 score, etc...
Figures:

Fig1.png Figure showing the basic mechanism to create and contribute to a given pool

Fig3.png Figure showing a crowded pool: the number of IDs connected to a pool can be used as an indicator of engagement

Fig4.png Figure showing a pool with few users connected: again, the number of IDs connected to a pool can be used as an indicator of engagement

Fig7.png Figure showing "temporal clustering". Ada contributed to 2 pools in a shorter amount of time respec to Tom: She has a higher chance of becoming an engaged user
