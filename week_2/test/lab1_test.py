import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week_2.preprocessing.simple_tokenizer import SimpleTokenizer
from week_2.preprocessing.regex_tokenizer import RegexTokenizer

if __name__ == "__main__":
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    text = "Hello, world!"
    simple_tokens = simple_tokenizer.tokenizer(text)
    regex_tokens = regex_tokenizer.tokenizer(text)
    print("Simple tokens:\n", simple_tokens)
    print("Regex tokens:\n", simple_tokens)