from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pandas.core.frame import DataFrame

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.representation.count_vectorizer import CountVectorizer
from src.models.text_classifier import TextClassifier

def get_data(data_path: str) -> DataFrame:
    """Get data

    Args:
        data_path (str): path to data

    Returns:
        Dataframe: the dataframe if success else None
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print("Error load data: ", e)

        return None
    
def main():
    data_path = "data\sentiments.csv"
    df = get_data(data_path)
    df["text"] = df["text"].str.lower()
    
    X_train, X_test, y_train, y_test = train_test_split(df["text"].values, df["sentiment"].values, test_size=0.2, random_state=42)
    
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer=tokenizer)
    classifier = TextClassifier(vectorizer=vectorizer)
    
    classifier.fit(X_train, y_train)
    print(classifier)
    predicts = classifier.predict(X_test)
    print(predicts[:20])
    print(y_test[:20])
    results = classifier.evaluate(y_test, predicts)
    acc, precision, recall, f1 = results["acc"], results["precision"], results["recall"], results["f1"]
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

if __name__ == "__main__":
    main()
