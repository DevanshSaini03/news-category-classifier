Here is a professionally formatted `README.md` file designed for a high-quality project submission. It covers the technical details of your "from-scratch" implementation and provides clear instructions for anyone reviewing your work.

---

# Sport vs. Politics Text Classifier (from Scratch)

## 📌 Project Overview

This project implements a robust text classification system to distinguish between **Sports** and **Politics** news articles. Unlike standard implementations that rely on `scikit-learn`, all core machine learning logic—including feature extraction and model optimization—was built from first principles using **Python 3**.

### Objectives:

* Perform binary classification on high-dimensional text data.
* Implement and compare **Probabilistic**, **Discriminative**, and **Instance-based** learning.
* Develop a custom **Bag of Words (BoW)** vectorization pipeline.

---

## 📂 Dataset Details

The system utilizes the **HuffPost News Category Dataset v3**.

* **Source:** [Kaggle - News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
* **Format:** Newline-delimited JSON.
* **Categories:** Filtered specifically for `SPORTS` and `POLITICS`.
* **Input Features:** Concatenated `headline` and `short_description`.

---

## 🛠️ Implementation Specs

### 1. Feature Engineering

* **Preprocessing:** Custom regex-based cleaning (lowercasing, punctuation removal).
* **Vectorization:** Manual implementation of a **Bag of Words** model that builds a vocabulary from the training set and transforms text into frequency-based feature vectors.

### 2. Models Implemented

* **Naive Bayes:** Uses log-likelihoods to prevent numerical underflow and **Laplace Smoothing** to handle out-of-vocabulary words.
* **Logistic Regression:** Built using **Batch Gradient Descent** with a manual Sigmoid activation function and iterative weight updates.
* **K-Nearest Neighbors (KNN):** Uses **Euclidean Distance** metrics and a majority voting system ().

---

## 📊 Performance Comparison

Based on a test split of 20%, the following accuracies were achieved:

| Model | Accuracy | Complexity |
| --- | --- | --- |
| **Naive Bayes** | **91.00%** | Low |
| **Logistic Regression** | **87.00%** | Medium |
| **KNN (k=5)** | **83.75%** | High |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with `numpy` and `pandas`.

```bash
pip install numpy pandas

```

### Installation & Execution

1. **Download the Dataset:** Obtain `News_Category_Dataset_v3.json` from Kaggle.
2. **Setup:** Place the JSON file in the same directory as the script.
3. **Run:**
```bash
python RollNumber_prob4.py

```



---

## ⚠️ Limitations

* **Context:** The Bag of Words model does not capture word order or semantic meaning (polysemy).
* **Scaling:** The KNN implementation is computationally intensive for very large datasets ( prediction time).
* **Vocabulary:** Accuracy is highly dependent on the quality and diversity of the training vocabulary.

---

## 📜 License

This project was developed for the **Natural Language Understanding (NLU)** course assignment. All model logic is original work.

---

**Next Step:** I've included the LaTeX math and tables to make it look great on GitHub or a PDF. Would you like me to add a section on how to tune the **Learning Rate** for the Logistic Regression part?
