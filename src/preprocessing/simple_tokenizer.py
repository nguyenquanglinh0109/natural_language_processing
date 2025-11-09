import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sys.path.append('..')
from src.core.interfaces import Tokenizer

import re
import string

class SimpleTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text):
        text = text.lower()
        pattern = r"([ ,.!?])"
        tokens = re.split(pattern, text)
        tokens = [i for i in tokens if i != '' and i != ' ']
        return tokens

