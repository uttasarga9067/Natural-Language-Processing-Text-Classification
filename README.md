# Natural-Language-Processing-Text-Classification

### Sentiment Analyisis of Movie Reviews

Classifying a text data from a Data Source which consists of Movie Reviews. The processing of Text Data is mandatory before we start applying Machine Learning Techniques on them.
We classified whether the Movie is having a Positiove or a negative rating by assigning them 1; if the rating is greater than 7 and 0 if the rating is less than 4. There are some unlabled data which I did not include in my Analysis.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/1.PNG)

The text_train is a list of length 25000, while I have printed the Reviews which consists of positive ratings(1). We now clean the data and remove the uncessary formatiing in order to make sure that these dont have any impact on our Machine Learning Model.

The Number of Positive and Negative ratings are the same; which is :

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/2.PNG)

The Motive behind this Analysis is we have to label a review as positive or negative; based on the text which are present in such reviews. We should now proceed to convert String Representation of such texts into Numeric Labels; in order to make our Models Work.

### Bag-Of-Words Representation

This Analysis helps us to check that in a document; which word appeared in every line of the document and assign them the label of 1 for appearence and 0 for absence. For example:


![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/3.PNG)

Here is the Representation of the above statement after processing:

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/4.PNG)

After applying the same on our dataset, we can see that it consists of 25000 lines; with 74849 words in them. Let us check the representation in much more detail.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/5.PNG)

Here, we can see that the first 20 features really dont make any sense rather than some numbers; which can be a part of a movie. Also, many words are of the same meaning; so counting them as different feature wont make sense for our analysis and modeling.
Before going into more further analysis; I trained the model based on my bag of words representation; and I receive 88% Accuracy in my Logistic Regression Model. The reason we choose this model is because it works well in binary representation of the Data.

Now, we tune the regularization parameter C via Cross-Validation.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/6.PNG)

We obtain an Accuracy of 89%, which is 1 percent improvement from our previous Model where there was no CV Assigned. After using the recent Model on Test Set, we get an Accuracy of 88% !!

Now, moving on to feature extraction, we can see that the words that Count vectorizer converts Upper Case letters to Lower Case and then we decide to use only those features that are present in minimum of 5 documents.

After this, we can see that the Number of Vocabulary is down to 27271 features and now, we look into the tokens present in these features.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/7.PNG)

Here, the words are different and are not repetative or having the same meaning. Now, we train our model on this Data that we extracted and see how it works.
We get an accuracy of 89% which is the same even if we removed atleast one-third of our data. This means that we have included those features which are already of much greater significance.

We can get rid of English Stop words, so that the words which dont play any significance but are still present many times in a corpus are removed.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/8.PNG)

We remove those Stop Words, which brings down our Vocabulary Bucket to 26966 and then we perform Grid Search again; which gives us 88% Acuuracy. This method usually helps in Small datasets, as we dont have enough features to be removed from the List.

### Term Frequency- Inverse Document Frequency:

I was in my Information Retrival Class where I first learned about this Method. This approach has a unique intuition; which gives maximum rating to those words or features, that appear a lot in a single document and minimum rating to those words which tend to appear in more documents. We use a Pipeline for this approach as this uses Statistical properties of the Data.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/9.PNG)

We get 89% Accuracy again; but the catch here is that Tf-Idf gives us the words which are of the least and most important in a text corpus.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/10.PNG)

Here, the words which are repeated in all documents most of the time are Stop words and hence, We can exclude those words from the Dataset.

Now, we can check which words put of the 27000 features are classified from our Logistic Regression model and we can see those analysis of the top 25 Largest and Smallest Coefficients.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/11.PNG)


### N-Grams for Analysis

N-Grams are used for text sentimental analysis, because we encounter such words which have a opposite meaning; when they are used together with some other word present in the document. For example, lets take 'good' and 'not good'. Both Words are of different meaning, but if we dont include bi-grams in our analysis; we wont be able to catch these words and hence our analysis can be biased which we dont want to happen on a large scale.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/12.PNG)

You can see in the above image that the sentence is now divided into bi-grams and then used for assigning bag-of-words represntation. This helps us in many ways which we will see below.

We are now using N-Grams with Tf-IDF vectorization technique, in order to assign rankings to those bag-of-words present in every document or in a single document in a corpus.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/13.PNG)

We get a score of 91%, with bi-grams and tri-grams helping us in a good manner. I have printed out a heatmap for those who are into numbers (Like Me) and have it segregated according to the Regularization parameters which are used in Cross Validation.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/14.PNG)

We can again check which features are the most decisive in our Logistic Regression model trained using N-Grams.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/15.PNG)

Here, we can see that Unigrams and Bi-Grams are changing the game and impacting our Model by a lot.
If I just look into the Tri-Grams present in the Dataset, I can see that those features are not of a greater significance in the Model.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/16.PNG)


### Some More Advanced Features of Tokenization, Stemming and Lemmatization

Stemming is a technique in which common suffixes of a word is dropped. This process identifies a word stem for each word present in the corpus. If the role of a particular word is taken into account in a sentence; for example if its a verb, or a noun and so on; then Lemmatization works efficiently and the word form which is generated is known as Lemma.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/17.PNG)

We start to perform Logistic Regression on both the type of the Datasets; first on the set of 27000 features and then on reduced data with lemmatization which is around 21000 features.

Since, Lemmatization algorithm works a bit slow, we will use just one percent of our training dataset, while having 99% of our test set in order to check the accuracy; which we get is around 1% more than the original Model. Lemmatization gives us a slight improvement and hence can be then used in order to train the Model on the Whole Training Set.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/18.PNG)

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/19.PNG)

### Topic Modeling and Document Clustering Techniques.

Latent Dirichlet Allocation helps us to find a group of words that appear together frequently. LDA helps us to get a picture that each document can be a mixture of different topics. We will use this technique in our Review Data set, in order to get the Topics for which our Dataset belongs.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/20.PNG)

The Topic Modeling helps us to judge various topics based on how the words are clustered. Topic 1 seems to be revolving around History and War; while Topic 3 consists of Reviews for TV & Web Series. Topic 6 gives us a picture about the Kids Genre and Anitmated Movies; which even adults like us love to watch and give them awards.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/21.PNG)

Below, We clustered aorund 100 Topics and here is the Analysis. Each topic consists of the Top 20 Words in 100-Topic Clustering Technique.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/22.PNG)

Choosing a single topic to check the Reviews associated with these topics, gives us the Idea that the intution I had for several topic was correct and my brain is now trained for such analysis.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/23.PNG)

TOPIC 45 consists of Music Reviews and here are the words clustered in TOPIC-45.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/24.PNG)

We can check the weight, which these topics get on the Overall Text Corpus.

![ScreenShot](https://github.com/uttasarga9067/Natural-Language-Processing-Text-Classification/blob/main/25.PNG)


To summarize Topic Models like LDA, we can use its techniques to capture certain labelled and unlabelled documents. Any kind of topics picked from the LDA should be taken and analysed completely in order to proceed which can be done by checking the Reviews belonging to that Model. I verified my own Intuition so there's that. This is helpful even if fewer training examples are available.

