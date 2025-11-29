from src.representation import CountVectorizer
from src.preprocessing import RegexTokenizer, SimpleTokenizer
from src.core.dataset_loaders import load_raw_text_data

def main():
    corpus = [
        "I love NLP, you love programming.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    regex_tokenizer = RegexTokenizer()
    simple_tokenizer  = SimpleTokenizer()

    count_vectorizer = CountVectorizer(regex_tokenizer)
    count_vectorizer.fit(corpus=corpus)
    print(corpus)
    print(count_vectorizer.vocab_)    

    count_vectors = count_vectorizer.transform(corpus)
    print(count_vectors)

def test_ud_dataset():
    data_path = "./data/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
    data = load_raw_text_data(data_path)
    sample_test = data[:500]
    regex_tokenizer = RegexTokenizer()
    simple_tokenizer  = SimpleTokenizer()

    count_vectorizer = CountVectorizer(regex_tokenizer)
    count_vectorizer.fit(corpus=sample_test)

    count_vectors = count_vectorizer.transform(sample_test)
    print(count_vectors)
    
    
if __name__ == "__main__":
    # a simple test
    main()
    
    # test_ud_dataset
    test_ud_dataset()