import re
from src.core.interfaces import Tokenizer
from typing import List

class RegexTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> str:
        pattern = r"\w+|[^\w\s]"
        tokens = re.findall(pattern, text)
        return tokens
    
def main():
    text = "kickers on my watchlist xide  soq pnk cpw bpz aj  trade method 1 or method 2, see prev posts'"
    tokenizer = RegexTokenizer()
    tokens = tokenizer.tokenize(text)
    print(tokens)
    
if __name__ == "__main__":
    main()