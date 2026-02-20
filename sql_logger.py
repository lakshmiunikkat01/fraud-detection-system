import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect("fraud_logs.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transaction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_amount REAL,
            predicted_class INTEGER,
            fraud_probability REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_transaction(amount, predicted_class, probability):
    conn = sqlite3.connect("fraud_logs.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO transaction_logs 
        (transaction_amount, predicted_class, fraud_probability, created_at)
        VALUES (?, ?, ?, ?)
    """, (amount, predicted_class, probability, datetime.now()))

    conn.commit()
    conn.close()


def show_logs():
    conn = sqlite3.connect("fraud_logs.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM transaction_logs")
    rows = cursor.fetchall()

    for row in rows:
        print(row)

    conn.close()


def fraud_summary():
    conn = sqlite3.connect("fraud_logs.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT predicted_class, COUNT(*) 
        FROM transaction_logs 
        GROUP BY predicted_class
    """)

    rows = cursor.fetchall()

    print("\nFraud Summary from DB:")
    for row in rows:
        print("Predicted Class:", row[0], "Count:", row[1])

    conn.close()