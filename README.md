# Spam-Email-Detection
📧 Spam Email Detection using Machine Learning

This project, developed during the second semester for the Basics of Machine Learning course, focuses on building a machine learning model to classify emails as spam or ham (non-spam). The aim is to enhance email security by detecting and filtering out malicious or unwanted messages.

🔍 Overview

Spam emails pose significant security risks, including phishing, malware, and data breaches. In this project, we created a spam detection system using classic NLP and ML techniques to automatically flag such emails.

📂 Dataset

Dataset used: email.csv (approx. 5000 emails)
Two categories: spam and ham
Preprocessed by removing nulls, duplicates, and unnecessary records


🧠 Features & Techniques

Text Preprocessing:
Tokenization
Stop word removal
Text normalization (lowercasing, punctuation removal)
Feature Engineering:
Bag-of-Words using CountVectorizer
Word frequencies and N-grams
Vectorization: CountVectorizer from Scikit-learn


🧪 Model Training

Algorithm Used: Logistic Regression with L1 regularization (liblinear solver)
Train-Test Split: 75% training, 25% testing
Libraries: NumPy, Pandas, Scikit-learn, NLTK, Matplotlib, Seaborn


📈 Evaluation Metrics

Metric	Value
Accuracy	98.29%
Precision	96.75%
Recall	89.75%
F1-Score	93.12%


🚀 Future Improvements

Improve robustness against spam evasion techniques
Integrate user feedback for active learning
Scale the model for real-time detection in large email systems
📎 References

Dataset Source: Kaggle Spam Dataset
Project Notebook: Colab Link


🛠 How to Run

pip install -r requirements.txt
python spam_detector.py
Note: Make sure the email.csv dataset is in the same directory.

📌 Conclusion

This project demonstrates how simple ML models can effectively classify spam emails with high accuracy, contributing to safer and more productive digital communication.
