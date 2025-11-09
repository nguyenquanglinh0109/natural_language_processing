import re
from week_2.core.interfaces import Tokenizer
from typing import List

class RegexTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenizer(self, text: str) -> str:
        pattern = r"\w+|[^\w\s]"
        tokens = re.findall(pattern, text)
        return tokens