from week_2.core.interfaces import Vectorizer
from week_2.preprocessing.regex_tokenizer import RegexTokenizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        # self.vocab_: Dict[str, int] = {}
        self.vocab = []
        super().__init__()

    def fit(self, corpus) -> None:
        text = ' '.join(corpus)
        tokens = RegexTokenizer().tokenizer(text)
        vocab = list(set(tokens)) + ["<UNK>"]
        return list(vocab)
    
    def transform(self, documents):
        # check vocab trước, khởi tạo [0] * vocab_size, count số lượng
        count_vectors = []
        self.vocab = self.fit(documents)
        for document in documents:
            document = [self.vocab.index(i) if i in self.vocab else len(self.vocab) for i in RegexTokenizer().tokenizer(document)]
            count_vectors.append(document)

        return count_vectors
    
    def fit_transform(self, corpus):
        count_vectors = []
        for document in corpus:
            document = [self.vocab.index(i) if i in self.vocab else len(self.vocab) for i in RegexTokenizer().tokenizer(document)]
            count_vectors.append(document)

        return count_vectors