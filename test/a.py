import sys
import os
import tokenize
from week_2 import RegexTokenizer, SimpleTokenizer

print("toan bo duong dan ma module se tim kiem")
print(sys.path)

print("duong dan tuyet doi den file hien tai")
print(__file__)

print("thu muc cha cua file hien tại")
print(os.path.dirname(__file__))

def main():
    print("hello")
    text = "Hello, world! This is a test."
    tokenizer = RegexTokenizer()
    print(tokenizer.tokenizer(text))

if __name__ == "__main__":
    main()