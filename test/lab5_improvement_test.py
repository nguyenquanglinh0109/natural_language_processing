from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.representation.count_vectorizer import CountVectorizer
from src.representation.word_embedder import WordEmbedder
from src.models.improve_text_classifier import ImproveTextClassifier
from typing import List
from collections import Counter

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

def get_uncommon_words(documents: List[str], count_threshold: int = 1) -> List[str]:
    regex_tokenizer = RegexTokenizer()
    all_words = [word for doc in documents for word in regex_tokenizer.tokenize(doc)]
    counter = Counter(all_words)
    uncommon_words = [word for word, count in counter.items() if count == count_threshold]
    
    return uncommon_words

def preprocess_text(documents: List[str], uncommon_words: List[str]) -> str:
    results = []
    for document in tqdm(documents, desc="Preprocessing documents"):
        tokens = RegexTokenizer().tokenize(document)
        for word in uncommon_words:
            tokens = [token.replace(word, "") for token in tokens]
        
        document = " ".join(tokens)
        results.append(document)

    return results

def main():
    data_path = r".\data\sentiments.csv"
    df = get_data(data_path)
    df["text"] = df["text"].str.lower()
    
    X_train, X_test, y_train, y_test = train_test_split(df["text"].values, df["sentiment"].values, test_size=0.2, random_state=42)
    
    # Preprocessing
    uncommon_words = get_uncommon_words(X_train, count_threshold=3)
    print(uncommon_words)
    
    print("Preprocessing text...")
    X_train = preprocess_text(X_train, uncommon_words)
    X_test = preprocess_text(X_test, uncommon_words)
    
    vectorizer = WordEmbedder("glove-wiki-gigaword-300")
    classifier = ImproveTextClassifier(vectorizer=vectorizer)
    
    print("Training model...")
    classifier.fit(X_train, y_train)
    print(classifier)
    
    print("Predicting...")
    predicts = classifier.predict(X_test)
    results = classifier.evaluate(y_test, predicts)
    acc, precision, recall, f1 = results["acc"], results["precision"], results["recall"], results["f1"]
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

if __name__ == "__main__":
    main()
    # data_path = r".\data\sentiments.csv"
    # df = get_data(data_path)
    
    # df["text"] = df["text"].str.lower()
    # documents = df["text"].values
    # uncommon_words = get_uncommon_words(documents)
    
    # cleaned_documents = preprocess_text(documents, uncommon_words)
    # print(cleaned_documents[:10])
    # print(uncommon_words)
