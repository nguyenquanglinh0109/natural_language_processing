from torch.utils.data import Dataset, DataLoader
import torch
from src.core.build_vocab import Vocabulary, get_data
from torch.nn.utils.rnn import pad_sequence


PAD_IDX = 0
UNK_IDX = 1


class POSDataset(Dataset):
    def __init__(self, sentences, word_to_idx, tag_to_idx):
        super().__init__()
        self.sentences = sentences
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]

        sentence_indices = [
            self.word_to_idx.get(token, UNK_IDX)
            for token, _ in sentence
        ]

        tag_indices = [
            self.tag_to_idx[tag]
            for _, tag in sentence
        ]
        
        return torch.tensor(sentence_indices), torch.tensor(tag_indices)
            

def custom_collate_fn(batch):
    sentences = [sentence for sentence, _ in batch]
    tags = [tag for _, tag in batch]
    
    # CHỖ NÀY ĐÃ SỬA — padding_index phải là PAD_IDX = 0
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=PAD_IDX)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=PAD_IDX)
    
    return padded_sentences, padded_tags
                    

def main():
    data_path = r"data\UD_English-EWT\UD_English-EWT\en_ewt-ud-train.conllu"

    # Load raw sentences
    sentences = get_data(data_path)

    # Vocabulary phải tạo PAD=0, UNK=1
    vocab = Vocabulary(data_path)
    vocab.build_index()   # đảm bảo build_index có thêm PAD/UNK
    
    word_to_idx = vocab.word_to_idx
    tag_to_idx = vocab.tag_to_idx

    dataset = POSDataset(sentences, word_to_idx, tag_to_idx)
    print(dataset[0]) 

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate_fn
    )

    for batch in loader:
        print(batch)
        break 
               
if __name__ == "__main__": 
    main()
