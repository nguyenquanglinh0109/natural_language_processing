import token
from week_2.representation.count_vectorizer import CountVectorizer
from week_2.preprocessing.regex_tokenizer import RegexTokenizer
from week_2.preprocessing.simple_tokenizer import SimpleTokenizer

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


if __name__ == "__main__":
    main()