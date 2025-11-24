# **Lab 2: Count Vectorization**
---
## Implement Bag of Words
### Implement
```
Implement Bag of Words with dictionary words get from corpus
Use transform to convert documents to vectors
```

### Run code 
```
py -m test.lab2_test.py
```

### Corpus
```
Document 1: "I love NLP, you love programming."
Document 2: "I love programming."
Document 3: "NLP is a subfield of AI."
```

### Vocabulary
```
{'<UNK>': 0, ',': 1, '.': 2, 'AI': 3, 'I': 4, 'NLP': 5, 'a': 6, 'is': 7, 'love': 8, 'of': 9, 'programming': 10, 'subfield': 11, 'you': 12}

==> Tạo tập từ điển từ corpus ban đầu, thêm token <UNK> đại diện cho các từ chưa từng xuất hiện trong corpus
```
### Bag of Words representation
```
bow_matrix = [
    # <UNK>, ,, ., AI, I, NLP, a, is, love, of, programming, subfield, you
    [  0,   1, 1,  0, 1,   1, 0,  0,    2,  0,          1,        0,   1],  # Doc 1
    [  0,   0, 1,  0, 1,   0, 0,  0,    1,  0,          1,        0,   0],  # Doc 2
    [  0,   0, 1,  1, 0,   1, 1,  1,    0,  1,          0,        1,   0]   # Doc 3
]

==> Từ nào xuất hiện sẽ được cộng thêm 1 đơn vị, không xuất hiện sẽ là 0.
```

## Problem: No
## Reference: No
## Pretrained Model: No