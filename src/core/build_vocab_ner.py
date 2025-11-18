import datasets
from typing import List, Dict

def transform_tag(tags: List[List[int]], tags_map: Dict[str, int]):
    results = []
    for tag in tags:
        results.append([tags_map[i] for i in tag])
    return results

def get_data(data_path: str, tags_map):
    try: 
        print("Loading dataset...")
        dataset = datasets.load_dataset(data_path)
        train_sentences = dataset["train"]["tokens"]
        train_tags = dataset["train"]["ner_tags"]
        
        val_sentences = dataset["validation"]["tokens"]
        val_tags = dataset["validation"]["ner_tags"]
        
        test_sentences = dataset["test"]["tokens"]
        test_tags = dataset["test"]["ner_tags"]
        
        train_tags = transform_tag(train_tags, tags_map)
        val_tags = transform_tag(val_tags, tags_map)
        test_tags = transform_tag(test_tags, tags_map)
    
    except Exception as e:
        print("Error load data: ", e)
        return None
    
    return train_sentences, train_tags, val_sentences, val_tags, test_sentences, test_tags
    

class Vocabulary:
    def __init__(self, data):
        self.data = data
        self.word_to_idx = {}
        self.tag_to_idx = {}
    
    def build_index(self):
        for sentence, tags in self.data:
            for word, tag in zip(sentence, tags):
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx) + 2
                if tag not in self.tag_to_idx:
                    self.tag_to_idx[tag] = len(self.tag_to_idx)
        
        self.word_to_idx["<UNK>"] = 1 
        self.word_to_idx["<PAD>"] = 0
        
        print("Successfully build vocabulary")
        print("Length of word_to_idx:", len(self.word_to_idx))
        print("Length of tag_to_idx:", len(self.tag_to_idx))
        
                            
def main():
    data_path = "lhoestq/conll2003"
    tags_map = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    train_sentences, train_tags, val_sentences, val_tags, test_sentences, test_tags = get_data(data_path, tags_map)
    data = list(zip(train_sentences, train_tags))
    
    vocab = Vocabulary(data)
    vocab.build_index()
    print(vocab.word_to_idx)
    print(vocab.tag_to_idx)
    
if __name__ == "__main__":
    main()




