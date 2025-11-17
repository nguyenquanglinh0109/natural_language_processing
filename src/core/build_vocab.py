import pyconll

def get_data(data_path: str):
    results = []
    conll_data = pyconll.load_from_file(data_path)
    for sentence in conll_data:
        sample = [(token.form, token.upos) for token in sentence]
        results.append(sample)
    
    return results

class Vocabulary:
    def __init__(self, data_path: str):
        self.data = get_data(data_path) if isinstance(data_path, str) else data_path
        self.word_to_idx = {}
        self.tag_to_idx = {}
    
    def build_index(self):
        for sentence in self.data:
            for word, tag in sentence:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx) + 1
                if tag not in self.tag_to_idx:
                    self.tag_to_idx[tag] = len(self.tag_to_idx)
        self.word_to_idx["<UNK>"] = 0 
        print("Len word_to_idx:", len(self.word_to_idx))
        print("Len tag_to_idx:", len(self.tag_to_idx))
        
                            
def main():
    data_path = r"data\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.conllu"
    conll_data = get_data(data_path)
    # print(conll_data[:1])
    vocab = Vocabulary(data_path)
    vocab.build_index()
    print(vocab.word_to_idx)
    print(vocab.tag_to_idx)
    
if __name__ == "__main__":
    main()




