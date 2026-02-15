import os
import re
import json
import random
import math
import numpy as np
import pandas as pd
from collections import Counter

class TextPreprocessor:
    def __init__(self):
        self.vocab = {}
        
    def clean(self, text):
        text = str(text).lower()
        return re.sub(r'[^a-z\s]', '', text).split()

    def fit_transform(self, docs):
        clean_docs = [self.clean(d) for d in docs]
        unique_words = set(word for doc in clean_docs for word in doc)
        self.vocab = {word: i for i, word in enumerate(sorted(list(unique_words)))}
        
        X = np.zeros((len(docs), len(self.vocab)), dtype=int)
        for i, doc in enumerate(clean_docs):
            for word in doc:
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1
        return X

    def transform(self, docs):
        clean_docs = [self.clean(d) for d in docs]
        X = np.zeros((len(docs), len(self.vocab)), dtype=int)
        for i, doc in enumerate(clean_docs):
            for word in doc:
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1
        return X

class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / n_samples
            token_counts = np.sum(X_c, axis=0) + 1
            self.likelihoods[c] = np.log(token_counts / np.sum(token_counts))

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                posteriors[c] = np.log(self.priors[c]) + np.sum(x * self.likelihoods[c])
            preds.append(max(posteriors, key=posteriors.get))
        return np.array(preds)

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.classes = []
        
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.classes = np.unique(y)
        pos_class = self.classes[1] 
        y_enc = np.where(y == pos_class, 1, 0)
        
        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            
            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y_enc))
            db = (1.0 / n_samples) * np.sum(y_pred - y_enc)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear)
        return np.array([self.classes[1] if p > 0.5 else self.classes[0] for p in y_pred])

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(dists)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            preds.append(Counter(k_labels).most_common(1)[0][0])
        return np.array(preds)

def load_data(filename):
    texts = []
    labels = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cat = item.get('category', '').upper()
                    
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

    texts, labels = load_data(filename)
    
    if len(texts) == 0:
        print("No valid data found in JSON.")
        return

    indices = np.arange(len(texts))
    random.shuffle(indices)
    
    if len(texts) > 2000:
        indices = indices[:2000]
        
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_raw_train = texts[train_idx]
    y_train = labels[train_idx]
    X_raw_test = texts[test_idx]
    y_test = labels[test_idx]
    
    vec = TextPreprocessor()
    X_train = vec.fit_transform(X_raw_train)
    X_test = vec.transform(X_raw_test)
    
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    nb_acc = np.mean(nb.predict(X_test) == y_test)
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")

    lr = LogisticRegression(lr=0.1, epochs=500)
    lr.fit(X_train, y_train)
    lr_acc = np.mean(lr.predict(X_test) == y_test)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    knn_acc = np.mean(knn.predict(X_test) == y_test)
    print(f"KNN Accuracy: {knn_acc:.4f}")

if __name__ == "__main__":
    main()