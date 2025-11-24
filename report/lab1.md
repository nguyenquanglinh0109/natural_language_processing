# **Lab 1: Tokenization**
---
## SimpleTokenizer and RegexTokenizer
### Implement
```
1. Define interface: Tokenizer with abstract method tokenize
Implement: SimpleTokenizer (split by [ ,.?]), RegexTokenizer (split by pattern: "\w+|[^\w\s]")
Evaluation: 
  + sample test: corpus = [
                    "Hello, world! This is a test.",
                    "NLP is fascinating... isn't it?",
                    "Let's see how it handles 123 numbers and punctuation!"
                ]
  + UD Englist EWT dataset

2. Define interface: Vectorizer with abstract method fit, transform, fit_transform
Implement: Bag of words
Evaluation: 
  + sample test: corpus = [
                    "Hello, world! This is a test.",
                    "NLP is fascinating... isn't it?",
                    "Let's see how it handles 123 numbers and punctuation!"
                ]
  + UD Englist EWT dataset
```

### Run code 
```
py -m test.lab1_test.py
```

### Result
```text
Doc1: Hello, world! This is a test.

Simple: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
Regex: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
==================================================

Doc2: NLP is fascinating... isn't it?

Simple: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
Regex: ['NLP', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
==================================================

Doc3: Let's see how it handles 123 numbers and punctuation!

Simple: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
Regex: ['Let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
==================================================


Nhận xét: 
  + SimpleTokenizer tách thành các token đơn giản với khoảng trắng và các dấu câu: let's -> let's
  + RegexTokenizer tách thành các token với biểu thức chính quy: Let's -> [Let, ', s]
```


## Problem: No
## Reference: No
## Pretrained Model: No
