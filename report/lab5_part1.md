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
| Word2Vec + Dense               | 0.82     | 0.83      | 0.82   | 0.6658   |
| Embedding (Pre-trained) + LSTM | 0.64     | 0.64      | 0.65   | 1.1718   |
| Embedding (Scratch) + LSTM     | 0.76     | 0.77      | 0.77   | 1.2057   |

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
Theo giả thuyết phương pháp hiện đại biểu diễn vector bằng embedding, mô hình phân loại với LSTM sẽ cho lại hiệu quả cao hơn so với pipeline TF-IDF + Logistic Regression. Tuy nhiên thực tế huấn luyện lại không phản ánh được điều đó, trong quá trình traning mô hình với LSTM cho kết quả trên tập train tương đối tốt cho thấy khả năng học tập của nó tuy nhiên lại mất đi tính tổng quát do quá fit vào tập train, khi đem ra đánh giá trên tập test hiệu suất bị giảm đi nhiều.
```

### Reference
[Sklearn](https://scikit-learn.org/stable/index.html) \
[Tensorflow](https://www.tensorflow.org/)
[Pytorch](https://pytorch.org/)

