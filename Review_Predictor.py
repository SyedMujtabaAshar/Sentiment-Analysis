import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load Dataset
# Replace with the path to your dataset file
# Assuming a CSV file with 'review' and 'sentiment' columns
# Sentiment: 1 (positive), 0 (negative)
url = "C:/Users/Ashar Master/Desktop/DevHubs Internship Tasks/Task 2/sentiment_analysis.csv"
data = pd.read_csv(url)


# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing
data['cleaned_review'] = data['text'].apply(preprocess_text)

# Feature Engineering: Convert text to numerical format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cleaned_review'])

# Labels
data['sentiment'] = data['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})
y = data['sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Prediction function
def predict_sentiment(review):
    # Preprocess the input review text
    processed_review = preprocess_text(review)

    # Vectorize the preprocessed review
    vectorized_review = vectorizer.transform([processed_review])

    # Predict the sentiment using the trained model
    prediction = model.predict(vectorized_review)[0]

    # Map the numerical prediction back to the corresponding sentiment label
    if prediction == 2:
        sentiment = "Positive"
    elif prediction == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Negative"
    
    return sentiment

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Allow the user to input a review at runtime
sample_review = input("Enter your review: ")

# Print the review and the predicted sentiment
print(f"Predicted Sentiment: {predict_sentiment(sample_review)}")