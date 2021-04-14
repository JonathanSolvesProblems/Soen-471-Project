## Team
Jonathan Andrei <br />
Andrew Ha <br />
Inés Gonzalez Pepe <br />

# Abstract

The aim of our project is to be able to predict the number of up-votes a user has on a post, based on the Parler messaging app. We consider the up-votes to be a representative of the user's engagement with others. 
We aim to determine which factors, such as keywords, type of user, time of day, etc. contribute to how popular a post will get. To this end, we intend to utilize a linear regression model, as well as a decision tree model 
to distinguish between the quality of their predictions and the internal data processing and manipulation by each model.

# Introduction

After the Capital riots on January 6th in the US, analyses were published online discussing links to rioters, based on leaked messages, hashtags and locations. This, in association with the discovery of an article detailing the 
creation of several different types of popularity predicting models trained on Reddit posts, led to the objective of creating a model that would accurately predict the number of upvotes (aka the popularity) on a Parler post. 
The dataset that will be used is composed of leaked messages and associated metadata from the alt-tech (uncensored applications and services developed by companies which tend to be favored by right-wing individuals) messaging app 
Parler and contains identifiers such as username, time of day, key words, etc. 

In line with the dataset analysis course project requirements of integrating two different techniques learned in class, we intend to create a linear regression model and a random forest model to compare the performance of the models 
by examining the quality of predictions as well as the internal data processing and data manipulation by each model. 

Our hypothesis is that the random forest model will have better performance than the linear regression model. We also believe that the most salient identifiers will be a combination of comments, usernames and key words used, as these 
features represent the celebrity of a user on Parler, hot topics that get people engaged and engagement itself. Moreover, the random forest model is useful in identifying the important features as it shifts the weights associated with 
each feature in order to determine the optimal ‘importance’ of each feature. It will be interesting to see the results in the training set with the adjustment of importance in features.

# Materials and Methods

The dataset of Parler users has been obtained from the free open source website Zenodo. The dataset contains posts and comments made by Parler 
users over the period of several months leading to the riots. This dataset is in .ndjson format, so we need to perform some preprocessing to 
convert our data into .csv format. In addition, we will be using Dask, Apache Spark, pandas and matplotlib to filter, clean and visualize our 
data and scikit-learn to build our ML model, do metrics analysis and potentially automated hyperparameter tuning. Lastly, with random forest, 
certain features will be preselected for the models to base their initial training on. Most of these features will require no processing 
(i.e. username of the poster, time of day, etc.), but the body of the post will need to be processed as we want to make it easy for the model 
to learn what the message says. To this end, we will use the word2vec library to help us convert the comments into a vector which is more 
readable by the model. For example, if posts with a certain word have a high up-vote count, then a word2vec of that word can help us classify 
future comments and identify similar words that will have the same effect. 
 
In order to achieve our objective, we need a dataset that has a high percentage of popular posts, but also has a decent range of posts with 
various ranges of popularity. To this end, we want our data distribution to resemble a bell curve and will be using the goodness of fit algorithm 
to select the best subsample possible from the raw Parler data. Further preprocessing is still tentative, but involves removal of partially empty 
or erroneous rows (some comment tallies amount to -1, which bears no meaning to us). For simplicity, only comments in English will be classified 
within our dataset initially. We may revisit them later on within our project to see if we can take our model further and use those foreign 
comments in an interesting way. Finally, another factor in our model’s performance that we intend to analyze is the effect of word frequency vs 
word embedding in regards to the processing of the Parler post’s content. 
 
In brief, we intend to determine first and foremost, the better model between linear regression and random forest, and second of all, we want to 
determine which features are most salient and how best to process our textual data.

