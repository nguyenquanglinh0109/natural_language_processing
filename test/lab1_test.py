import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import SimpleTokenizer, RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

def main():
    # ... (your tokenizer imports and instantiations) ...
    dataset_path = "./data/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()
    
    # Take a small portion of the text for demonstration
    sample_text = raw_text[:500] # First 500 characters
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")

if __name__ == "__main__":
    print("=======Simple test=======")
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    corpus = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
    for text in corpus:
        print(text)
        print(simple_tokenizer.tokenize(text))
        print(regex_tokenizer.tokenize(text))
        print("=" * 50)

    print("=======Ud english test=======")
    main()
