import pandas as pd

def preprocess_data(path):
    df = pd.read_csv(path)

    print("Shape:", df.shape)

    print("\nClass Distribution:")
    print(df["Class"].value_counts())

    print("\nFraud Percentage:")
    print((df["Class"].value_counts(normalize=True) * 100))

    print("\nAverage Transaction Amount by Class:")
    print(df.groupby("Class")["Amount"].mean())

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y