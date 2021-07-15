# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & Classification

## Problem statement
The objective of this project is to collect posts from two subreddits, using Reddit's API and then use NLP to train a classifier on which subreddit a given post came from. Data are to be gathered and prepared using the `requests` library. Two models are to be created and compared and one of the two models must be a Bayes classifier.


## Executive summary
#### Background
Reddit is an American social news aggregation, web content rating, and discussion website. Registered members submit content to the site such as links, text posts, images, and videos, which are then voted up or down by other members. Posts are organized by subject into user-created boards called "communities" or "subreddits".

#### Objective
The goal of this project is to collect posts from two subreddits, using Reddit's API and then use NLP to train a classifier on which subreddit a given post came from.

#### Model selection
For this project, the two models selected are **Logistic Regression** and **Multinomial Naive Bayes** and evaluation is based on mean cross-validated score of accuracy of the best estimator.

#### Data collection
To get started, posts are scraped from https://www.reddit.com/r/keto/ and https://www.reddit.com/r/stopdrinking/ using the `requests` library. 994 posts are scraped from the first subreddit, out of which, 742 posts are unique. All 995 posts from the second are unique bringing the total of unique posts to 1737 out of 1989 posts scraped. The two subreddits are carefully chosen for the subtle similarities (healthy, positivity etc.) between them while being on different subjects altogether. Choosing subreddits of two totally different subjects is to ensure there is variance between posts so our models can pick up the differences and learn from there to make distinction and the subtle similarities make it so that posts are not so different that distinction can be made immediately without difficulties making this a meaningful yet challenging project. Between every request, an intentional 2 to 6 seconds wait is generated to make it look more 'natural' so we can get more unique posts. Posts scraped are made into data frames and saved as csv files.

#### Data cleaning and EDA
Null and blank values in data can affect the accuracy of prediction so these are the first things that are checked for in data cleaning. Upon inspection, no null or blank are observed in either dataset. Next, only columns (features) that will be useful in making the prediction will be chosen. Useful columns 'title' and 'selftext' which will be analysed and column 'name' which will be referenced for duplicates are then extracted to form new data frames. Following on, duplicate rows are dropped by referencing 'name' column while 'title' and 'selftext' are merged to form a new column 'message'. A new column 'target' is made to identify which subreddit each post came from. Once all these are performed on both data sets, they are merged into a single data frame.

#### Preprocessing
Steps taken in preprocessing function:
1. Remove Url.
2. Remove non-letters.
3. Convert to lower case, split into individual words.
4. Remove stopwords.
5. Remove subreddit and words closely related to subreddit.

<p>Url are removed because they may contain the subreddit which will affect the accuracy of the prediction by directly telling the model which subreddit the post came from and other websites like Youtube or Twitter which has no relevance to the prediction. Non-letters like numbers and punctuations are removed because they don't serve any purpose in the modeling. Words are split into individual words and converted to lower case so they can be analysed individually and also to pave the way for them to be fitted into the transformers. Stop words which are common English words, which do not add much meaning and can safely be ignored without sacrificing the meaning of the sentence, are removed by using NLTK stopword to filter them out. Finally, subreddit and words that are closely related to subreddit are removed for the same first reason as removing Url. Steps taken in this preprocessing phase are to make the posts less biasful, more relevant and into a format that can be used for modeling. The processed messages are then stored back into the dataframe, ready for the next step: Modeling.</p>

#### Modeling
For this project, the two models selected are **Logistic Regression** and **Multinomial Naive Bayes** and two transformers selected are **Count Vectorizer** and **TFIDF Vectorizer**. Data is split in a training and a test set with a ratio of 75/25. Since this is a binary classification, baseline accuracy is determined by simply predicting the majority class which is 57.24% in this case.

#### Model: Logistic regression
Logistic regression is a statistical model that, in its basic form, uses a logistic function to model a binary dependent variable to find a relationship between features and probability of a particular outcome. In this case, the relationship between the posts and probability of being in one of the subreddits.
#### Score (base)
| Model (base)                              	| Train score 	| Test score 	|
|--------------------------------------------	|:-----------:	|:----------:	|
| Logistic regression with Count Vectorizer  	|     1.0     	|   0.9609   	|
| Logistic regression with  TFIDF Vectorizer 	|    0.9854   	|   0.9540   	|

#### Score (gridsearch)
| Model (gridsearch)                         	| Train score 	| Test score 	|
|--------------------------------------------	|:-----------:	|:----------:	|
| Logistic regression with Count Vectorizer  	|    0.9946   	|   0.9609   	|
| Logistic regression with  TFIDF Vectorizer 	|    1.0    	|   0.9586   	|

#### Analysis
A base (no tuning) logistic regression model is built with base (no tuning) transformers count vectorizer and tfidf vectorizer and the scores are shown in the Score (base) table above. First thoughts that come to mind are accuracies are high and there seem to be a little overfitting judging from the 100% accuracy in one of the training scores. 

<p>Count vectorizer counts the number of times each word occurs in the entire text and TFIDF vectorizer, explained simply, measures the importance of a word. In hope to improve accuracy of the model and reduce overfitting, a gridsearch is done on the model by tuning 3 parameters max_df, max_features and ngram_range on the transformer and the C parameter on the model. Max_df allows the transformer to ignore words that have a document frequency strictly higher than the set threshold thus ignoring words that appear in too many posts which either mean it is a common word or it is closely related to the subreddit. Max_features limits the the number of features (vocabulary) that the transformer will learn considering only top features ordered by word occurrence. Ngram_range sets the number of words that forms a term to be used in the analysis. C parameter controls the regularization strength of the model in a manner where a smaller C increases the strength which will create simple models which underfit the data.</p>

<p>The gridsearch only managed to improve the test score on the model with TFIDF vectorizer slightly and stayed exactly the same on the model with count vectorizer.</p>
    
#### Model: Naive bayes
Naive bayes is a powerful algorithm based on applying Bayes theorem with a strong(naive) assumption, that every feature is independent of the others, in order to predict the category of a given sample. In this case, it is used to predict the subreddit given the posts. 
#### Score (base)
| Model (base)                       	| Train score 	| Test score 	|
|------------------------------------	|:-----------:	|:----------:	|
| Naive bayes with Count Vectorizer  	|    0.9869   	|   0.9724   	|
| Naive bayes with  TFIDF Vectorizer 	|    0.9685   	|   0.9356   	|

#### Score (gridsearch)
| Model (base)                       	| Train score 	| Test score 	|
|------------------------------------	|:-----------:	|:----------:	|
| Naive bayes with Count Vectorizer  	|    0.9800   	|   0.9770   	|
| Naive bayes with  TFIDF Vectorizer 	|    0.9777   	|   0.9494   	|

#### Analysis
Same procedure as the previous model. As shown in the Score (base) table, Naive bayes model with count vectorizer scored really well with high accurracy and small gap between training and testing score. 

<p>Parameters used to tune the transformers remained the same and laplace smoothing is introduced to handle the problem of zero probability in Naive bayes model by tuning the alpha parameter. Overall accuracy improved in both cases as a result of the gridsearch.</p>

#### Conclusion and recommendations
First of all, the accuracy of every model is quite high. What could have happened could be the subtle similarities between the 2 subreddit could be a little too subtle than I have expected thus making them too different and easy for the models to distinguish between them. Both models improved only very slightly or even no improvement in terms of accuracy probably because the base scores are already really high and there aren't much room for improvements. The naive assumption of Naive bayes model is that all features are conditionally independent which works exceptionally well in this project especially with the steps taken in preprocessing which further reduced the dependency of one word to another thus emerging as the best model in this project with a test score of 0.977.

<p>Overall, objectives of this project have been well met by successfully scraping 1989 posts out of an intended 2000 and getting 1737 unique posts for analysis. 2 models are successfully fitted, tuned and tested to eventually yield a production model being Naive bayes with count vectorizer at a test score of 0.977.</p>
