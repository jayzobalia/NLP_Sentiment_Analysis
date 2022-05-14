################################## IMPORTS ##################################
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

################################## Loading Dataset ##################################

df = pd.read_csv('train.csv')

################################## Pre-processing ##################################

# Replacing all the null values
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df.dropna(how='all', axis=0, inplace=True)
df = df[:][:10]


# 1) Removing all the @user tags from all tweets.
# 2) Removing all special characters and symbols (keeping only words and hashtags).
# 3) Removing all the words having less than 3 letters.
# 4) Trimming all the extra while spaces.

def clean_tweet(str1, str2):
    for i in range(len(df['tweet'])):
        tweet = df['tweet'][i]
        df['tweet'][i] = re.sub(str1, ' ', tweet)
        temp3 = df['tweet'][i]
        df['tweet'][i] = re.sub(str2, ' ', temp3)
        temp2 = df['tweet'][i]
        df['tweet'][i] = re.sub(r'\b\w{1,3}\b', '', temp2)
        temp1 = df['tweet'][i]
        df['tweet'][i] = re.sub('\\s+', ' ', temp1)


clean_tweet('@[\w]*', '[^a-zA-Z#]')

################################## Tokenization ##################################

df['tweet'] = df['tweet'].apply(lambda x: x.split())

##################################   STEMMING   ##################################
# Using the Porter stemmer as it is to be more efficient for
# tweets than the Lancaster Stemmer which seemed more
# aggressive for the tweets in laymen terms and words.

stemmer = nltk.stem.PorterStemmer()
df['tweet'] = df['tweet'].apply(lambda x: [stemmer.stem(word) for word in x])

# Joining the tokenized tweets
for i in range(len(df['tweet'])):
    df['tweet'][i] = " ".join(df['tweet'][i])

all_words = " ".join(sentence for sentence in df['tweet'][df['label'] == 0])

##################################   Visualization   ##################################

# Using the wordcloud library to visualize the
# frequency of thw words in all the tweets
wordcloud = WordCloud(width=800, height=500, random_state=1000000, max_font_size=100).generate(all_words)

# ploting the graph
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

##################################   Saving the processed dataframe   ##################################

# Doing so to not have to process the data again and again after this point whenever we run the code
df = pd.DataFrame(dict)
df.to_csv('processed_tweets.csv')

##################################   Loading the processed dataframe   ##################################
new_df = pd.read_csv('processed_tweets.csv', names=['id', 'label', 'tweet'])

##################################   Getting HastTags    ##################################
# In this part of the code, we get all the hashtags in 2 lists,
# hashtags of positive tweets and negative tweets.

hashtags_positive = []
hashtags_negative = []
count_pos = []
count_neg = []
abc, bbc = 0, 0
for i in range(len(new_df['tweet'])):
    temp_list = []
    if new_df['label'][i] == 0:
        temp_list = re.findall(r"#(\w+)", str(new_df['tweet'][i]))
        count_pos.append(len(temp_list))
        count_neg.append(0)
    if new_df['label'][i] == 1:
        temp_list = re.findall(r"#(\w+)", str(new_df['tweet'][i]))
        count_neg.append(len(temp_list))
        count_pos.append(0)

# Creating 2 new columns in the dataframe to store the count of
# Positive and negative hashtags in each tweet, which we will
# further use for training the model.

new_df['positive_tags'] = count_pos
new_df['negative_tags'] = count_neg

tweets = []
for i in range(len(new_df['tweet'])):
    tweets.append(str(new_df['tweet'][i]))


##################################   Tweet(word) vectorizer    ##################################

# Using the CountVectorizer to converts the stemmed words in to
# array for the machine to recognize.

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(tweets)
bow = bow.toarray()

##################################   Model Training    ##################################

# Creating 2 different models, replicating 2 different
# approaches of performing sentiment analysis. The 2 models
# trained have different attributes (one has vectorized words
# and second one has number of positive and negative hashtags
# in the tweet). Both of them have different accuracy.

# Eventhough the accuracy of the model with number of positive and
# negative hashtags is greater, it isn't suitable as getting labelled
# dataset of number of positive and negative hashtags is not likely.

x = new_df[['positive_tags', 'negative_tags']].values
y = new_df['label'].values

x_train1, x_test1, y_train1, y_test1 = train_test_split(bow, y, random_state=42, test_size=0.2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, random_state=42, test_size=0.2)

model1 = LogisticRegression()
model1.fit(x_train1, y_train1)

model2 = LogisticRegression()
model2.fit(x_train2, y_train2)

pred1 = model1.predict(x_test1)
pred2 = model2.predict(x_test2)


##################################   Confusion MATRIX    ##################################
print("\n" + "*" * 50)
print("Accuracy using CountVectorizer", (accuracy_score(y_test1, pred1)) * 100)
print("Accuracy using HashTags", (accuracy_score(y_test1, pred2)) * 100)
print("\n" + "*" * 50)

print("Confusion Matrix for MODEL 1")
print(metrics.confusion_matrix(y_test1, pred1))
print(metrics.classification_report(y_test1, pred1))
print("\n" + "*" * 50)

print("Confusion Matrix for MODEL 2")
print(metrics.confusion_matrix(y_test2, pred2))
print(metrics.classification_report(y_test2, pred2))
