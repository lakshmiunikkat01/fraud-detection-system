#  Fraud Detection System Using Machine Learning

##  Overview

This project builds a Fraud Detection System using machine learning techniques on highly imbalanced financial transaction data.

The goal is to detect fraudulent transactions while minimizing financial loss and reducing false positives.

This simulates how real fintech companies detect suspicious credit card activity.

---

##  Business Problem

Fraudulent transactions represent less than 0.2% of total transactions, making accuracy an unreliable metric.

Key challenges:
- Missing fraud = direct financial loss
- Too many false alerts = poor user experience

This project focuses on:
- Handling extreme class imbalance
- Optimizing fraud recall
- Estimating financial loss
- Logging predictions into SQL database

---

##  Dataset

- Credit Card Fraud Detection Dataset
- 284,807 total transactions
- 492 fraudulent transactions (~0.17%)

Highly imbalanced binary classification problem.

---

##  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SMOTE (Imbalanced-learn)
- SQLite (SQL logging)

---

##  Project Structure
fraud_detection_system/
│
├── data/
├── models/
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── sql_logger.py
│
├── main.py
├── requirements.txt


---

##  Methodology

### 1️ Exploratory Data Analysis
- Fraud distribution analysis
- Fraud percentage calculation
- Average transaction comparison

### 2️ Handling Imbalance
- Applied SMOTE oversampling
- Controlled oversampling to reduce overfitting

### 3️ Model Training
Compared:
- Logistic Regression
- XGBoost

### 4️ Business Metrics Evaluation
Evaluated using:
- Precision
- Recall
- Confusion Matrix
- Fraud Recall
- Financial Loss Estimation

---

##  Model Performance Example

| Model | Fraud Recall | Fraud Precision |
|--------|--------------|----------------|
| Logistic Regression | 86.7% | 33% |
| XGBoost | 82.6% | 83.5% |

Tradeoff:
- Logistic → High Recall, Lower Precision
- XGBoost → Balanced Performance

---

##  Financial Impact Simulation

Each missed fraud assumed to cost ₹50,000.

Example:
- Missed Frauds: 13
- Estimated Financial Loss: ₹650,000

This demonstrates real-world business decision impact.

---

##  SQL Logging

Predictions stored in SQLite database including:
- Transaction Amount
- Predicted Class
- Fraud Probability
- Timestamp

