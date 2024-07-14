import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Importing Data set
df = pd.read_csv("mail_data.csv")
#print all the data in the csv file
#print('Data in the csv file:\n',df)

data= df.where(pd.notnull(df), '')
#print('Top 5 results:\n', data.head(5))

#Information of data 
#data.info()

data.loc[data['label']=='spam', 'label',] =0
data.loc[data['label']=='ham', 'label',] =1

x= data['message']
#print(x)

y= data['label']
#print(y)

#training the model

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=(0.2), random_state= 3) 
#print(x.shape)
#print(x_train.shape)
#print(x_test.shape)
"""
print(y.shape)
print(y_train.shape)
print(y_test.shape)

"""
# Creating TfidfVectorizer
feature_extraction =TfidfVectorizer(min_df =1, stop_words='english', lowercase=True)
x_train_feature = feature_extraction.fit_transform(x_train)
x_test_feature = feature_extraction.transform(x_test)

#setting up y value to integer 
y_train = y_train.astype(int)
y_test = y_test.astype(int)

#print(x_train)
#print(x_train_feature)

#creating  logistic regression 
model = LogisticRegression()

#Training the model
model.fit(x_train_feature, y_train)

#training the prediction model
prediction_on_training_data = model.predict(x_train_feature)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
#print('accuracy on training data :' , accuracy_on_training_data)

#testing the prediction model
prediction_on_testing_data = model.predict(x_test_feature)
accuracy_on_testing_data = accuracy_score(y_test, prediction_on_testing_data)
#print('accuracy on testing data :' , accuracy_on_testing_data)

#input_your_mail = ['Please review and respond promptly.']
input_your_mail = ['Hi! See you in the evening']
input_data_feature = feature_extraction.transform(input_your_mail)


# Making prediction on input data
prediction = model.predict(input_data_feature)
print('Prediction: 0 indicates spam mail, 1 indicates ham mail.\nThe mail is:', prediction)

if(prediction[0]==1):
    print('Ham Mail')
else:
    print('Spam Mail')

#Note: It will predict one mail at a time
