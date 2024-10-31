import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
import nltk
from nltk.corpus import stopwords
import re
import time

# Download stopwords
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('emails.csv')

# Explore the dataset
print("\nInformation about emails")
print(data.info(),"\n")
print("\nTotal Entries", data['spam'].value_counts(), "\n")

# Pre-processing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()               # Convert to lowercase
    text = text.split()               # Split into words
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(text)             # Join back to a string

print("Processing....\n")
data = data.sample(200)  # Take a sample of 200 rows
data['cleaned_text'] = data['text'].apply(preprocess_text)

print("Selection....\n")
# Feature extraction
X = CountVectorizer().fit_transform(data['cleaned_text'])
y = data['spam']

# Feature selection (Select top k features)
k = 200  # Number of top features to select
X_selected = SelectKBest(chi2, k=k).fit_transform(X, y)

# Train-test split
print("Splitting....\n")
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

print("-----------------------------------------------\n\n")
print("\t\t\" Gaussian Naive Bayes: \"")
# Train and time the Gaussian Naive Bayes model
print("Training Gaussian Naive Bayes....\n")
start_time = time.time()
gnb_model = GaussianNB()  # Instantiate the Gaussian Naive Bayes model
gnb_model.fit(X_train.toarray(), y_train)  # Train the model
gnb_training_time = time.time() - start_time

# Make predictions for Gaussian Naive Bayes
print("Predicting with Gaussian Naive Bayes....\n")
start_time = time.time()
gnb_predictions = gnb_model.predict(X_test.toarray())  # Make predictions
gnb_prediction_time = time.time() - start_time

# Evaluate the Gaussian Naive Bayes model
print("Confusion Matrix:\n", confusion_matrix(y_test, gnb_predictions))
gnb_accuracy = accuracy_score(y_test, gnb_predictions)
print("Accuracy:", gnb_accuracy)
print("Error Rate:", 1 - gnb_accuracy)
print("Training Time:", gnb_training_time)
print("Prediction Time:", gnb_prediction_time)
print("Classification Report:\n", classification_report(y_test, gnb_predictions))

print("-----------------------------------------------\n\n")
print("\t\t\" Multinomial Naive Bayes: \"")
# Train and time the Multinomial Naive Bayes model
print("Training Multinomial Naive Bayes....\n")
start_time = time.time()
mnb_model = MultinomialNB()  # Instantiate the Multinomial Naive Bayes model
mnb_model.fit(X_train, y_train)  # Train the model
mnb_training_time = time.time() - start_time

# Make predictions for Multinomial Naive Bayes
print("Predicting with Multinomial Naive Bayes....\n")
start_time = time.time()
mnb_predictions = mnb_model.predict(X_test)  # Make predictions
mnb_prediction_time = time.time() - start_time

# Evaluate the Multinomial Naive Bayes model
print("Confusion Matrix:\n", confusion_matrix(y_test, mnb_predictions))
mnb_accuracy = accuracy_score(y_test, mnb_predictions)
print("Accuracy:", mnb_accuracy)
print("Error Rate:", 1 - mnb_accuracy)
print("Training Time:", mnb_training_time)
print("Prediction Time:", mnb_prediction_time)
print("Classification Report:\n", classification_report(y_test, mnb_predictions))

print("-----------------------------------------------\n\n")
print("\t\t\" Decision Tree (as J48): \"")
# Train and time the Decision Tree model (as a proxy for J48)
print("Training Decision Tree (as J48)....\n")
start_time = time.time()
dt_model = DecisionTreeClassifier()  # Instantiate the Decision Tree model
dt_model.fit(X_train, y_train)  # Train the model
dt_training_time = time.time() - start_time

# Make predictions for Decision Tree
print("Predicting with Decision Tree (as J48)....\n")
start_time = time.time()
dt_predictions = dt_model.predict(X_test)  # Make predictions
dt_prediction_time = time.time() - start_time

# Evaluate the Decision Tree model
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Accuracy:", dt_accuracy)
print("Error Rate:", 1 - dt_accuracy)
print("Training Time:", dt_training_time)
print("Prediction Time:", dt_prediction_time)
print("Classification Report:\n", classification_report(y_test, dt_predictions))
