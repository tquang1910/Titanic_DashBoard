import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Load Titanic training, test, and gender_submission datasets."""
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    gender = pd.read_csv("data/gender_submission.csv")
    return train, test, gender

def preprocess(df):
    """Clean and encode Titanic dataset."""
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')

    le = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df