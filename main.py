from src.preprocess import preprocess_data
from src.train import train_model

if __name__ == "__main__":
    X, y = preprocess_data("data/creditcard.csv")
    train_model(X, y)