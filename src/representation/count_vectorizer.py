from src.core.interfaces import Vectorizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        super().__init__()
        self._tokenizer = tokenizer
        self.vocab_: dict[str, int] = {}  # từ python 3.9 có thể dùng trực tiếp list, dict thay vì import từ Typing (list, Dict)

    def fit(self, corpus) -> None:
        words = set()
        for text in corpus:
            tokens = self._tokenizer.tokenize(text)
            words.update(tokens)
        
        words = sorted(list(words))
        words.insert(0, "<UNK>")
        self.vocab_ = {token: i for i, token in enumerate(words)}
    
    def transform(self, documents) -> list[list[int]]:
        if not self.vocab_:
            raise ValueError("Don't have vocab, use fit() to create")
        vectors = []

        for document in documents:
            count_vector = [0] * len(self.vocab_)
            tokens = self._tokenizer.tokenize(document)
            for token in tokens:
                token_idx = self.vocab_.get(token, self.vocab_["<UNK>"])
                count_vector[token_idx] += 1
            vectors.append(count_vector)
        return vectors
    
    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)