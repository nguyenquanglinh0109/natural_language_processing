import re
from week_2.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenizer(self, text):
        pattern = r"\w+|[^\w\s]"
        tokens = re.split(pattern, text)
        return tokens