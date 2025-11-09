import gensim
from src.preprocessing.regex_tokenizer import RegexTokenizer
import numpy as np
from typing import List
from tqdm import tqdm

class WordEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = gensim.downloader.load(model_name)
        self.n_dim = len(self.model["the"])

    def get_vector(self, word: str):
        try:
            return self.model[word]
        except KeyError:
            # print(f"Word {word} not found in model {self.model_name}")
            return np.random.rand(self.n_dim)
            
    def get_similarity(self, word1: str, word2: str):
        return self.model.similarity(word1, word2)
    
    def get_most_similarity(self, word: str, top_n: int = 10):
        return self.model.most_similar(word, topn=top_n)
    
    def embed_document(self, document: str):
        tokens = RegexTokenizer().tokenize(document)
        vectors = []
        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)

        if len(vectors) == 0:
            return [0] * self.n_dim

        return np.mean(vectors, axis=0)
    
    def transform(self, documents: List[str]):
        results = []
        for document in tqdm(documents, desc="Transforming documents"):
            results.append(self.embed_document(document))
        return results