import os
import re
import json
import random
import math
import numpy as np
import pandas as pd
from collections import Counter

class TextPreprocessor:
    """Handles text cleaning and transformation into a Bag of Words matrix."""
    def __init__(self):
        self.vocab = {}
        
    def clean(self, text):
        # Convert to string, lowercase, and remove all non-alphabetic characters
        text = str(text).lower()
        return re.sub(r'[^a-z\s]', '', text).split()

    def fit_transform(self, docs):
        # Build the vocabulary from a training set and return its sparse count matrix
        clean_docs = [self.clean(d) for d in docs]
        unique_words = set(word for doc in clean_docs for word in doc)
        # Create a mapping of word to index sorted alphabetically
        self.vocab = {word: i for i, word in enumerate(sorted(list(unique_words)))}
        
        X = np.zeros((len(docs), len(self.vocab)), dtype=int)
        for i, doc in enumerate(clean_docs):
            for word in doc:
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1
        return X

    def transform(self, docs):
        # Transform unseen docs using the previously built vocabulary
        clean_docs = [self.clean(d) for d in docs]
        X = np.zeros((len(docs), len(self.vocab)), dtype=int)
        for i, doc in enumerate(clean_docs):
            for word in doc:
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1
        return X

class NaiveBayes:
    """Probabilistic classifier based on feature frequency."""
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            # Prior P(c) = samples in class / total samples
            self.priors[c] = X_c.shape[0] / n_samples
            # Log Likelihood with Laplace Smoothing (+1)
            token_counts = np.sum(X_c, axis=0) + 1
            self.likelihoods[c] = np.log(token_counts / np.sum(token_counts))

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                # Summing log likelihoods for Bag of Words features
                posteriors[c] = np.log(self.priors[c]) + np.sum(x * self.likelihoods[c])
            preds.append(max(posteriors, key=posteriors.get))
        return np.array(preds)

class LogisticRegression:
    """Discriminative model using gradient descent optimization."""
    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr # Learning rate
        self.epochs = epochs # Number of optimization iterations
        self.weights = None
        self.bias = 0
        self.classes = []
        
    def sigmoid(self, z):
        # Logistic function to map input into a (0,1) probability
        z = np.clip(z, -250, 250) # Avoid numerical overflow in exp
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.classes = np.unique(y)
        # Binary encoding: map one class to 1 and the other to 0
        pos_class = self.classes[1] 
        y_enc = np.where(y == pos_class, 1, 0)
        
        # Training loop for Gradient Descent
        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            
            # Derivative calculation for weights and bias
            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y_enc))
            db = (1.0 / n_samples) * np.sum(y_pred - y_enc)
            
            # Update parameters in direction of steepest descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear)
        # Classify based on 0.5 threshold
        return np.array([self.classes[1] if p > 0.5 else self.classes[0] for p in y_pred])

class KNN:
    """Lazy learner that classifies based on spatial distance."""
    def __init__(self, k=5):
        self.k = k # Number of neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN simply stores the training data without explicit parameter learning
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            # Vectorized Euclidean Distance calculation
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Sort distances and get indices of the 'K' nearest points
            k_indices = np.argsort(dists)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            # Majority vote among the neighbors
            preds.append(Counter(k_labels).most_common(1)[0][0])
        return np.array(preds)

def load_data(filename):
    """Loads and filters a JSON dataset for specific news categories."""
    texts = []
    labels = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cat = item.get('category', '').upper()
                    
                    # Focus only on Politics and Sports
                    if cat in ['POLITICS', 'SPORTS', 'SPORT']:
                        txt = item.get('headline', '') + " " + item.get('short_description', '')
                        if 'SPORT' in cat:
                            labels.append('SPORTS')
                        else:
                            labels.append('POLITICS')
                        texts.append(txt)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return np.array([]), np.array([])
                
    return np.array(texts), np.array(labels)

def main():
    filename = 'News_Category_Dataset_v3.json'
    
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    # Data Acquisition and shuffling
    texts, labels = load_data(filename)
    if len(texts) == 0:
        print("No valid data found in JSON.")
        return

    indices = np.arange(len(texts))
    random.shuffle(indices)
    
    # Use a subsample for performance during the evaluation phase
    if len(texts) > 2000:
        indices = indices[:2000]
        
    # 80/20 Train-Test split
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_raw_train, X_raw_test = texts[train_idx], texts[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    # Text vectorization phase
    vec = TextPreprocessor()
    X_train = vec.fit_transform(X_raw_train)
    X_test = vec.transform(X_raw_test)
    
    # Model 1: Naive Bayes
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    nb_acc = np.mean(nb.predict(X_test) == y_test)
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")

    # Model 2: Logistic Regression
    lr = LogisticRegression(lr=0.1, epochs=500)
    lr.fit(X_train, y_train)
    lr_acc = np.mean(lr.predict(X_test) == y_test)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # Model 3: KNN
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    knn_acc = np.mean(knn.predict(X_test) == y_test)
    print(f"KNN Accuracy: {knn_acc:.4f}")

if __name__ == "__main__":
    main()                  
    
