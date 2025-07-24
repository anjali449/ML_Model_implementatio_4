#Machine Learning Model Implementation : Spam detection

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

#Create a simple dataset
data = {
    'text': [
        'Win money now by texting WIN to 12345',
        'Hi, how are you doing today?',
        'Congratulations! You have won a free vacation',
        'Can we meet for lunch tomorrow?',
        'Urgent! Call us now to claim your prize',
        'Let’s catch up this weekend',
        'You have been selected for a lottery reward',
        'Please review the meeting agenda and respond'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

#Convert to DataFrame
df = pd.DataFrame(data)

#Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
