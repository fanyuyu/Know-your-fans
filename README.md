# Know-your-fans
(Insight Data Science Project)

## Introduction
The goal of this project is to understand users' engagement on Facebook pages of a large media company and to identify superfans. The company has more than 30 Facebook channel pages, which are mainly for shows or networks. 

There are three types of user engagements recorded: 
* Like a post
* Comment on a post
* Share a post

Besides, below are some basic information about the dataset:
* The number of posts on each pages varies from 1 to more than 500.
* The datasets include user IDs, page IDs, and post IDs, and users' engagement in one month. 
* Specifically, data contains information on whether a user liked/commented/shared a post on a page, and if so, the timestamp of the engagement.

## Exploratory Data Analysis

EDA was performed to understand users' engagement on the page, post, and user levels. 

Below are some insights from EDA:
1. Some pages are much more popular. They had more than 10,000+ engaged users in one month.
2. Majority of the users' engagement type is “like”, especially for the most popular pages.
3. There are a small proportion of users who had more than one type of engagement for each page.
4. There are an even smaller proportion of users who engaged across pages.

## Dataset Preprocessing

Data were subsetted based on page and time. The identification of superfans were conducted on each page separately because only a small proportion of users engaged across pages. Data were subsetted by time for the purpose of algorithm validation.

## Feature Engineering

Feature engineering was conducted to extract features in order to cluster users into different groups regarding their engagement pattern and active level. 13 features were considered for clustering analysis. They account both features for each type of engagement and features for the interrelationship between different types of engagements. 
For example, since posts that were shared involved users "shared by" and "shared to", *network analysis* was conducted to extract centrality measures, out-degree, closeness, authority, and hub, to measure users' performance in the sharing network.

## Clustering Analysis (using K-means and Gaussian Mixture)

Two cluster algorithms, K-means and Gaussian mixture, were implemented. The primary aim for clustering is to identify superfans who are actively engaged on the page. 

The selected algorithm were applied on the validation set, and results showed that the cluster of users who were identified as active users on the training set were relatively more active than other clusters on the validation sets.

## PCA-based dimensionality reduction
PCA was used to reduce the dimension of the feature matrix and the clusters were plotted against the first two dimensions of the projection matrix.


## Implementation
Libraries used:
* python 2.7
* pandas (to interact with dataset)
* numpy (for scientific computing)
* networkx (for generating network graph)
* sklearn (for kmeans, Gaussian mixture, PCA)
* plotly (for interactive visualization)


The files contains the main functions to call subroutines to run the pipeline in python2.7. 

The pipeline, including data cleaning, merge, subsetting, feature engineering, clustering, PCA, and validation, is automated for each page.
