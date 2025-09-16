from week_2.representation.count_vectoreizer import CountVectorizer
from week_2.preprocessing.regex_tokenizer import RegexTokenizer

def main(corpus):
    regex_tokenizer = RegexTokenizer()
    count_vectorizer = CountVectorizer(regex_tokenizer)
    count_vectorizer.fit(corpus=corpus)

    count_vectors = count_vectorizer.transform(corpus)
    return count_vectors


if __name__ == "__main__":
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    count_vectors = main(corpus=corpus)
    print(count_vectors)