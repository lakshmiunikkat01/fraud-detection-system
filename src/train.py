from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import os
import joblib

from sql_logger import init_db, log_transaction


def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    tn, fp, fn, tp = cm.ravel()

    fraud_recall = tp / (tp + fn)
    fraud_precision = tp / (tp + fp)

    print("Fraud Recall:", fraud_recall)
    print("Fraud Precision:", fraud_precision)

    avg_fraud_amount = 50000
    financial_loss = fn * avg_fraud_amount

    print("Missed Frauds:", fn)
    print("Estimated Financial Loss: ₹", financial_loss)

    # Log first 5 predictions
    for i in range(5):
        log_transaction(
            amount=50000,
            predicted_class=int(y_pred[i]),
            probability=float(y_prob[i])
        )


def train_model(X, y):

    init_db()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE Distribution:")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print(dict(zip(unique, counts)))

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_resampled, y_train_resampled)
    evaluate_model(log_model, X_test, y_test, "Logistic Regression")

    # XGBoost
    xgb_model = XGBClassifier(eval_metric='logloss')
    xgb_model.fit(X_train_resampled, y_train_resampled)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    print("\nModel Comparison Completed: Logistic vs XGBoost based on Fraud Recall and Financial Loss")

    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(xgb_model, "models/fraud_model.pkl")
    print("\nBest model saved to models/fraud_model.pkl")