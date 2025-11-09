from src.core.dataset_loaders import load_sentences
from gensim.models import Word2Vec
from src.preprocessing import RegexTokenizer
from typing import List

def train_model(sentences: List[str], save_path: str = "./results/word2vec_model.bin"):
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4, sg=1)   
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
def load_model(data_path: str):
    model = Word2Vec.load(data_path)
    return model

def main():
    # ... (your tokenizer imports and instantiations) ...
    dataset_path = "./data/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt"
    print("Loading dataset...")
    sentences = load_sentences(dataset_path)
    print(sentences[:5])
    tokenizer = RegexTokenizer()
    sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    
    save_path = "./results/word2vec_model.bin"
    
    print("Training model...")
    train_model(sentences, save_path)
    
    print("Loading model...")
    w2v_model = load_model(save_path)
    print(w2v_model.wv["forces"])
    
if __name__ == "__main__":
    main()