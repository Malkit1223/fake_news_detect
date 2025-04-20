import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to train the model
def train_model(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Handle missing values by filling with 'text'
    df = df.fillna("text")

    # Separate features and labels
    X = df["text"]
    y = df["label"]

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('classifier', PassiveAggressiveClassifier(max_iter=50))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Separate the vectorizer and model for saving
    vectorizer = pipeline.named_steps['tfidf']
    model = pipeline.named_steps['classifier']

    # Evaluate the model
    evaluate_model(model, vectorizer, X_test, y_test)

    # Save the model and vectorizer
    save_model(vectorizer, model)
    print("Model training complete and saved.")

# Function to evaluate the model
def evaluate_model(model, vectorizer, X_test, y_test):
    # Transform the test data using the vectorizer
    X_test_tfidf = vectorizer.transform(X_test)

    # Make predictions on the test data
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(accuracy * 100, 2)}%")

    # Confusion Matrix to show performance on FAKE and REAL categories
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

# Function to save the trained model and vectorizer
def save_model(vectorizer, model):
    # Save the vectorizer
    with open('Vetorizer_model.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print("Vectorizer saved as 'Vetorizer_model.pkl'.")

    # Save the model
    with open('finalized_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved as 'finalized_model.pkl'.")

# Function to load the trained model and vectorizer
def load_model():
    # Load the vectorizer
    with open('Vetorizer_model.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    # Load the model
    with open('finalized_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    return vectorizer, model