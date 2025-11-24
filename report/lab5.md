# **Lab 5: Token classification**#
## 5.1 Pytorch introduction
### Implement
```
1. Tensor: 
  + create tensor
  + tensor operation
  + index, slicing
  + view, reshape

2. Autograd

3. First model pytorch with nn.Module
```

### Run code
```
run: src/notebook/lab5_pytorch_intro.ipynb
```

### Result
#### Operation
```
X data: 
 tensor([[1, 2],
        [3, 4]])

X total: 
 tensor([[2, 4],
        [6, 8]])

X mul: 
 tensor([[ 5, 10],
        [15, 20]])

X matmul: 
 tensor([[ 5, 11],
        [11, 25]])
```

#### Simple model
```
Input tensor 
 tensor([1, 5, 0, 8])
Input shape: torch.Size([4])
Output shape: torch.Size([4, 2])
Output: tensor([[-0.0332,  0.4619],
        [ 0.0808,  0.3076],
        [ 0.1161,  0.2583],
        [ 0.1682,  0.1902]], grad_fn=<AddmmBackward0>)
```

#### Explain
```
Embedding matrix: 
  tensor([[ 1.4451,  0.8564,  2.2181],
          [ 0.5232,  0.3466, -0.1973],
          [-1.0546,  1.2780, -0.1722],
          [ 0.5238,  0.0566,  0.4263],
          [ 0.5750, -0.6417, -0.4976],
          [ 0.4747, -2.5095,  0.4880],
          [ 0.7846,  0.0286,  0.6408],
          [ 0.5832,  0.2191,  0.5526],
          [-0.1853,  0.7528,  0.4048],
          [ 0.1785,  0.2649,  1.2732]], requires_grad=True)

Input: [1, 5, 0, 8] -> Lần lượt lấy ra các hàng tương ứng với index, các hàng này chính là vector biểu diễn cho từ.

Output:
  tensor([[-0.0332,  0.4619],
        [ 0.0808,  0.3076],
        [ 0.1161,  0.2583],
        [ 0.1682,  0.1902]], grad_fn=<AddmmBackward0>)

==> Đầu ra có kích thước là (4,2) tương ứng với 4 tokens ở phía trên, mỗi từ được biểu diễn bởi 1 vector hai chiều do được đi qua nn.Linear(3, 2)
```

## 5.2 Text classification with Pytorch
### Implement
```
1. TF-IDF + Logistic Regression

2. Word2Vec + Dense

3. Embedding Pre-trained + LSTM

4. Embedding from scratch + LSTM

5. Evaluate and Comparison
```

### Run code
```
run: src/notebook/lab5_rnn_text_classification.ipynb
```

### Result
```
1. Hwu dataset
| Mô hình                        | F1-score | Precision | Recall | Loss     |
|--------------------------------|----------|-----------|--------|----------|
| TF-IDF + Logistic Regression   | **0.84** | 0.85      | 0.83   | N/A      |
| Word2Vec + Dense               | 0.38     | 0.38      | 0.40   | 2.0383   |
| Embedding (Pre-trained) + LSTM | 0.45     | 0.47      | 0.46   | 1.8702   |
| Embedding (Scratch) + LSTM     | 0.00     | 0.00      | 0.02   | 4.1512   |

2. Some samples
Texts: 
  [
    "can you remind me to not call my mom",
    "is it going to be sunny or rainy tomorrow",
    "find a flight from new york to london but not through paris"
  ]

Ground true:['reminder_create', 'weather_query', 'flight_search']

Logistic Regression:
['calendar_set' 'weather_query' 'general_negate']

Word2vec + Dense:
['email_query' 'weather_query' 'email_sendemail']

Embedding (Pretrained) + LSTM:
['takeaway_query' 'weather_query' 'social_post']

Embedding (Scratch) + LSTM:
['alarm_set' 'alarm_set' 'alarm_set']

```

### Explain
```
TF-IDF + Logistic Regresion cho kết quả tốt nhất trong cả 4 phương pháp (Scratch dang lỗi) với F1-score là 0.84. 
Word2Vec + Dense cho F1-score là 0.38, Embedding (Pre-trained) + LSTM cho F1-Score là 0.46 cho thấy sự cải thiện hiệu suất tương đối đáng kể của LSTM so với việc chỉ tính trung bình của các vector.
==> Việc sử dụng trung bình để tạo đại diện cho một câu có thể đã làm các phần thông tin bị hoà trộn vào nhau do đó khi xây dựng mô hình khiến hiệu quả tương đối thấp, thay vào đó sử dụng LSTM cho thấy một cải thiện từ 0.38 lên 0.46
```

### Problem
```
Theo giả thuyết phương pháp hiện đại biểu diễn vector bằng embedding, mô hình phân loại với LSTM sẽ cho lại hiệu quả cao hơn so với pipeline TF-IDF + Logistic Regression. Tuy nhiên thực tế huấn luyện lại không phản ánh được điều đó.

Mô hình Embedding Scratch đang gặp lỗi dù sử dụng toàn bộ input của phương pháp 3 Embedding Pretrained
```

### Reference
[Sklearn](https://scikit-learn.org/stable/index.html) \
[Tensorflow](https://www.tensorflow.org/)
[Pytorch](https://pytorch.org/)


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

