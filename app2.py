import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import streamlit as st 

# Load datasets
real = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

# Label the datasets
real['label'] = 1
fake['label'] = 0

# Combine datasets
news = pd.concat([real, fake], axis=0)

# Drop unnecessary columns
news = news.drop(['title', 'subject', 'date'], axis=1)

# Shuffle the dataset
news = news.sample(frac=1).reset_index(drop=True)

# Text cleaning function
def cleantext(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

# Apply text cleaning
news['text'] = news['text'].apply(cleantext)

# Split data into features and labels
x = news['text']
y = news['label']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Vectorization
vectorfunc = TfidfVectorizer()
xv_train = vectorfunc.fit_transform(x_train)
xv_test = vectorfunc.transform(x_test)

# Logistic Regression model
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("Logistic Regression Score:", LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr))

# Decision Tree Classifier
DCT = DecisionTreeClassifier()
DCT.fit(xv_train, y_train)
pred_dct = DCT.predict(xv_test)
print("Decision Tree Score:", DCT.score(xv_test, y_test))
print(classification_report(y_test, pred_dct))

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(xv_train, y_train)
pred_rfc = rfc.predict(xv_test)
print("Random Forest Score:", rfc.score(xv_test, y_test))
print(classification_report(y_test, pred_rfc))

# Gradient Boosting Classifier
# gbc = GradientBoostingClassifier()
# gbc.fit(xv_train, y_train)
# pred_gbc = gbc.predict(xv_test)
# print("Gradient Boosting Score:", gbc.score(xv_test, y_test))
# print(classification_report(y_test, pred_gbc))

# Function to output label
def output_label(n):
    return "Genuine News" if n == 1 else "Fake News"

# Manual Testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_test = pd.DataFrame(testing_news)
    new_test["text"] = new_test["text"].apply(cleantext)
    new_xv_test = vectorfunc.transform(new_test["text"])
    pred_lr = LR.predict(new_xv_test)
    #pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    return f" \n LR Prediction: {output_label(pred_lr[0])} \n RFC Prediction: {output_label(pred_rfc[0])}"

# Streamlit app
import streamlit as st
st.title("Fake News Detector")
news_input = st.text_input("Enter news article")
if st.button("Check News"):
    st.write(manual_testing(news_input))
