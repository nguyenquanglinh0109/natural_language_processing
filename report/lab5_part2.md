# **Lab 5: Token classification**#
## 5.3 Token classification with Pytorch (Pos tagging)
### Implement
```
1. Build class Vocabulary (src/core/build_vocab.py)
  + function: build_index -> word_to_idx, tag_to_idx

2. Dataset, DataLoader (src/loader/pos_dataloader.py)
  + POSDataset
  + custom_collate_fn: padding to max_len in a batch

3. Build RNN model for pos tagging
```

### Run code
```
Run: py -m test.lab5_pos_tagging.py
```

### Trained model
```
model: trained_model/pos_tagging_model.pth
```

### Result
```
1. UD English EWT
Test dataset:
  + Loss: 0.3205 
  + Acc: 0.9572

2. Sample test
Text: "Transformers acts as the model definition framework for SOTA machine learning models in text, computer vision, audio, video, and multimodal model, for both inference and training."

Ouput: [('Transformers', 'PROPN'), ('acts', 'PROPN'), ('as', 'PROPN'), ('the', 'DET'), ('model', 'NOUN'), ('definition', 'VERB'), ('framework', 'PROPN'), ('for', 'PROPN'), ('SOTA', 'PROPN'), ('machine', 'PROPN'), ('learning', 'VERB'), ('models', 'NUM'), ('in', 'ADP'), ('text,', 'PROPN'), ('computer', 'NOUN'), ('vision,', 'PROPN'), ('audio,', 'PROPN'), ('video,', 'PROPN'), ('and', 'PROPN'), ('multimodal', 'PROPN'), ('model,', 'PROPN'), ('for', 'PROPN'), ('both', 'PROPN'), ('inference', 'PROPN'), ('and', 'PROPN'), ('training.', 'PROPN')]
```

### Explain
```
1. Build vocab: Xây dựng bộ từ vựng với word_to_idx (ánh xạ từ sang chỉ số), tag_to_idx (ánh xạ nhãn sang idx)

2. PosDataset, Dataloader định nghĩa hàm **custom_collate_fn** để đưa tất cả các vector trong một batch về cùng một kích thước khi training

3. SimpleRNNForTokenClassification: Sử dụng hidden state từ LSTM tại mỗi token để dự đoán đầu ra pos tagging. Kết quả trả về là tuple (text, pos)
```

### Problem
```
Không gặp vấn đề gì
```

### Reference
[Pytorch](https://pytorch.org/)

## 5.4 Token classification with Pytorch (Ner)
### Implement
```
1. Build class Vocabulary (src/core/build_vocab_ner.py)
  + function: build_index -> word_to_idx, tag_to_idx
  + (<PAD>: 0, <UNK>: 1)
2. Dataset, DataLoader (src/loader/ner_dataloader.py)
  + NERDataset 
  + custom_collate_fn: padding to max_len in a batch

3. Build RNN model for ner
```

### Run code
```
Run: py -m test.lab5_ner_tagging.py
```

### Trained model
```
model: trained_model/ner_model.pth
```

### Result
```
1. UD English EWT
Test dataset:
  + Loss: 0.5824 
  + Acc: 0.2718

2. Sample test
Sample test: "VNU University is located in Hanoi"  

[('VNU', 'B-PER'), ('University', 'I-ORG'), ('is', 'O'), ('located', 'O'), ('in', 'O'), ('Hanoi', 'B-LOC')]
```

### Explain
```
1. Build vocab: Xây dựng bộ từ vựng với word_to_idx (ánh xạ từ sang chỉ số), tag_to_idx (ánh xạ nhãn sang idx)

2. NERDataset, Dataloader định nghĩa hàm **custom_collate_fn** để đưa tất cả các vector trong một batch về cùng một kích thước khi training

3. SimpleRNNForTokenClassification: Sử dụng hidden state từ LSTM tại mỗi token để dự đoán đầu ra ner. Kết quả trả về là tuple (text, ner)
```

### Problem
```
Không gặp vấn đề gì
```

### Reference
[Pytorch](https://pytorch.org/)

