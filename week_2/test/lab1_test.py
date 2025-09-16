import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week_2.preprocessing.simple_tokenizer import SimpleTokenizer
from week_2.preprocessing.regex_tokenizer import RegexTokenizer

if __name__ == "__main__":
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    corpus = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
    for text in corpus:
        print(text)
        print(simple_tokenizer.tokenizer(text))
        print(regex_tokenizer.tokenizer(text))
        print("=" * 50)
