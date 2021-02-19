## Team
Jonathan Andrei <br />
Andrew Ha <br />
Inés Gonzalez Pepe <br />

# Abstract

The aim of our project is to be able to predict the number of up-votes a user has on a post, based on the Parler messaging app. We consider the up-votes to be a representative of the user's engagement with others. 
We aim to determine which factors, such as keywords, type of user, time of day, etc. contribute to how popular a post will get. To this end, we intend to utilize a linear regression model, as well as a decision tree model 
to distinguish between the quality of their predictions and the internal data processing and manipulation by each model.

# Introduction

After the Capital riots on January 6th in the US, analyses were published online discussing links to rioters, based on leaked messages, hashtags and locations. This in association with the discovery of an article detailing the 
creation of several different types of popularity predicting models trained on Reddit posts led to the objective of creating a model that would accurately predict the number of upvotes (aka the popularity) on a Parler post. 
The dataset that will be used is composed of leaked messages and associated metadata from the alt-tech (uncensored applications and services developed by companies which tend to be favored by right-wing individuals) messaging app 
Parler and contains identifiers such as username, time of day, key words, etc. 

In line with the dataset analysis course project requirements of integrating two different techniques learned in class, we intend to create a linear regression model and a random forest model to compare the performance of the models 
by examining the quality of predictions as well as the internal data processing and data manipulation by each model. Moreover, random forest will help us select the importance of features to use, as the importance of weights can be 
shifted. It will be interesting to see the various accuracies in the training set with the adjustment of importance in features.

Our hypothesis states that the most salient identifiers will be a combination of comments, usernames and key words used, as these features represent the celebrity of a user on Parler, hot topics that get people engaged and engagement 
itself.

Moreover, the article’s justification in categorizing importance between features will be adopted within our project to help determine the most significant attributes. The features that have the most influence on an up-vote 
will be chosen, such as body for example, which contains the content of the post. In addition, we will be analyzing the reaction ratio, based on whether the body was empty and the user only included a photo or a link to an article.

# Materials and Methods

The dataset of Parler users has been obtained from the free open source website Zenodo. The dataset contains posts and comments made by Parler users over the period of several months leading to the Insurrection. 
We will be focusing our analysis on posts that were made by users. This dataset is in .ndjson format, so we need to perform some preprocessing to convert our data into .csv format. In addition, we will be using various Python 
libraries to help us achieve our end goal. In order to filter and clean our data we intend to use the Dask and Apache Spark libraries, however, to build our ML model we will use scikit-learn. More specifically, scikit-learn’s linear 
regression and random forest classifiers will be utilized for building the model, while the metrics analysis provided by scikit-learn will be used for post-processing performance analysis. Lastly, we will use matplotlib to visually 
represent the data. We will also experiment with hyperparameters in the random forest.

In order to achieve our objectives, we need a dataset that has a high percentage of popular posts, but also has a decent range of posts with various ranges of popularity. To this end, we want our data distribution to resemble a bell 
curve and will be using the goodness of fit algorithm to select the best subsample possible from the raw Parler data. Further preprocessing is still tentative, but involves removal of partially empty or erroneous rows (some comment 
tallies amount to -1, which bears no meaning to us). Finally, certain features will be preselected for the models to base their initial training on. Most of these features will require no processing (i.e. username of the poster, time 
of day, etc.), however the body of the post will be converted to a vector as we want to make it easy for the model to take into account what the message says. We will also be using the word2vec library to help us determine synonyms 
within the comments. If posts with similar words have a high up-vote count, then it can help us classify future comments with similar-meaning words. Furthermore, we also intend to test model performance first on word frequency and 
second on word context. Moreover, we will be using Jupyter notebook to smoothly display the results. For simplicity, only comments in English will be classified within our dataset initially. We may revisit them later on within our 
project to see if we can take our model further and use those foreign comments in an interesting way.
