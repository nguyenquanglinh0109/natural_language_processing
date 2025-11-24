
# **Lab6: Transformer**
## Implement
```
1. Masked Language Model
2. Next Token Prediction
3. Sentence Representation
```

## Run code
```
py -m test.lab6_intro_transformers
```

## Result
### Masked Language Token
```
Câu gốc: Hanoi is the <mask> of Vietnam.
Dự đoán: 'capital' với độ tin cậy: 0.9354
 -> Câu hoàn chỉnh: Hanoi is the capital of Vietnam.
Dự đoán: 'center' với độ tin cậy: 0.0251
 -> Câu hoàn chỉnh: Hanoi is the center of Vietnam.
Dự đoán: 'heart' với độ tin cậy: 0.0109
 -> Câu hoàn chỉnh: Hanoi is the heart of Vietnam.
Dự đoán: 'centre' với độ tin cậy: 0.0032
 -> Câu hoàn chỉnh: Hanoi is the centre of Vietnam.
Dự đoán: 'city' với độ tin cậy: 0.0030
 -> Câu hoàn chỉnh: Hanoi is the city of Vietnam.

==> Mô hình dự đoán đúng 'capital'
==> Các mô hình Encoder-only như BERT phù hợp đặc biệt tốt cho tác vụ dự đoán từ <mask> vì kiến trúc encoder của Transformer được thiết kế để hiểu ngữ nghĩa hai chiều. Do đó nếu trong câu có 1 từ bị che, mô hình sẽ sử dụng được ngữ cảnh trước và sau để dự đoán từ phù hợp nhất.
```

### Next token prediction
```
Câu mồi: 'The best thing about learning NLP is'

Văn bản được sinh ra:
The best thing about learning NLP is that it offers a way to understand and interpret how people communicate, including nonverbal cues, emotions, and language. It's not just about the language itself, but how people use it. The field is rapidly growing, and it's becoming more and more important in our daily lives. It's a powerful tool for professionals who work with language, including those in education, business, and healthcare.

NLP can be used in multiple ways, including in the workplace, in education, and in healthcare.

==> Kết quả sinh ra tương đối hợp lý.
==> Các mô hình Decoder-only phù hợp và đặc biệt tốt cho tác vụ sinh văn bản vì kiến trúc decoder của Transformer được thiết kế chính xác cho việc dự đoán token tiếp theo theo chiều trái → phải.
```

### Sentence Representation
```
sentences = [
    "This is a sample sentence.",
    "Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads."           
]

{'input_ids': 
  + tensor([[  101,  2023,  2003,  1037,  7099,  6251,  1012,   102,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0],
          [  101,  3086,  2015, 15871,  2044,  1996,  3086,  3730, 17848,  1010,
          2109,  2000, 24134,  1996, 18215,  2779,  1999,  1996,  2969,  1011,
          3086,  4641,  1012,   102]])}

Vector biểu diễn của câu:
tensor([[-0.0639, -0.4284, -0.0668,  ..., -0.1753, -0.1239,  0.3197],
        [-0.1905, -0.3300,  0.3282,  ...,  0.0159, -0.0221, -0.1398]])

Kích thước của vector: torch.Size([2, 768])

==> Kích thước của vector biểu diễn là 768, tương ứng với hidden dim của BERT
==> Sử dụng attention mask vì có phần padding để tạo ra độ dài câu khác nhau, do đó cần mask để bỏ qua phần được thêm vào.
```

## Pretrained model
```
Masked Language Token: xlm-roberta-base
Next token prediction: Qwen/Qwen3-0.6B
Sentence Representation: bert-base-uncased
```

## Problem: No

## Reference
[PySpark](https://spark.apache.org/docs/latest/api/python/reference/index.html)
[Hugging Face](https://huggingface.co/)
