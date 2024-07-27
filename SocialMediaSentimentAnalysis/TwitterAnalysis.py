#For social media analysis i took 'Twitter'  
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

data = pd.read_csv('Twitter sentiments.csv')
# Display the first few rows of the DataFrame
print(data.head())
#Display all data
#print(data) 

#datatype info
data.info()

#setting the countvectorizer 
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(data['tweet'])

feature_names = bow_vectorizer.get_feature_names_out()
#words used in tweets
print("\nFeature names:\n\n", feature_names)
#Array in int 0 -ve, 1 neutral, 2 +ve
print("\nFirst tweet vector:\n\n", bow[0].toarray())
#counting of +ve, -ve and neutral words in tweet data
print(data['sentiment'].value_counts())

# visualization of freuently used words

all_words = "".join([sentence for sentence in data['tweet']])
wordcloud= WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

#plotting the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Freuently used random words from tweets')
plt.show()

# visualization of freuently used positive  words 
all_words = "".join([sentence for sentence in data['tweet'][data['sentiment']=='positive']])
wordcloud= WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

#plotting the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Freuently used  positive words from tweets')
plt.show()

# visualization of freuently used negattive  words
all_words = "".join([sentence for sentence in data['tweet'][data['sentiment']=='negative']])
wordcloud= WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

#plotting the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Freuently used negative words from tweets')
plt.show()

# Preprocess the tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tweet = tweet.lower()  # Convert to lowercase
    return tweet

data['tweet'] = data['tweet'].apply(preprocess_tweet)

# Vectorizing the tweets
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['tweet']).toarray()
y = data['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the sentiments
y_pred = model.predict(X_test)

# Compute the F1 Score and Accuracy
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
print(f'F1 Score (macro): {f1}')
print(f'Accuracy Score: {accuracy}')

# Plotting the results
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results['Actual'] = results['Actual'].map({2: 'positive', 1: 'neutral', 0: 'negative'})
results['Predicted'] = results['Predicted'].map({2: 'positive', 1: 'neutral', 0: 'negative'})

#plotting the graph
plt.figure(figsize=(10, 6))
sns.countplot(x='Actual', hue='Predicted', data=results)
plt.title('Actual vs Predicted Sentiments')
plt.xlabel('Actual Sentiment')
plt.ylabel('Count')
plt.legend(title='Predicted Sentiment')
plt.show()
