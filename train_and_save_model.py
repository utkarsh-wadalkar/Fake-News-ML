
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Train Logistic Regression
lr = LogisticRegression()
lr.fit(xv_train, y_train)

# Train Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(xv_train, y_train)

# Train Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(xv_train, y_train)

# Save the models and vectorizer to a dictionary
model_data = {
    'lr': lr,
    'gbc': gbc,
    'rfc': rfc,
    'vectorizer': vectorizer
}

# Save to model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Models and vectorizer saved to model.pkl")