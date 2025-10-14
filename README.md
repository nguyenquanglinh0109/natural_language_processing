# **TIMELINE**
+ [<ins>Week 1 (09/09/25)</ins>](#week-1)
+ [<ins>Week 2 (15/09/25)</ins>](#week-1)
    + [SimpleTokenizer and RegexTokenizer](#simpletokenizer-and-regextokenizer)
    + [Bag of Words](#bag-of-words-representation)

# **Week 1**

---

# **Week 2**
---
## SimpleTokenizer and RegexTokenizer
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
```

## Implement Bag of Words
### Corpus
```
Document 1: "I love NLP, you love programming."
Document 2: "I love programming."
Document 3: "NLP is a subfield of AI."
```

### Vocabulary
```
{'<UNK>': 0, ',': 1, '.': 2, 'AI': 3, 'I': 4, 'NLP': 5, 'a': 6, 'is': 7, 'love': 8, 'of': 9, 'programming': 10, 'subfield': 11, 'you': 12}
```
### Bag of Words representation
```
bow_matrix = [
    # <UNK>, ,, ., AI, I, NLP, a, is, love, of, programming, subfield, you
    [  0,   1, 1,  0, 1,   1, 0,  0,    2,  0,          1,        0,   1],  # Doc 1
    [  0,   0, 1,  0, 1,   0, 0,  0,    1,  0,          1,        0,   0],  # Doc 2
    [  0,   0, 1,  1, 0,   1, 1,  1,    0,  1,          0,        1,   0]   # Doc 3
]
```

# **Week 3**
Spark\
Lập trình hàm


# **Week 4**
## Visualize work embedding with PCA
### Word2Vec
![Visualize Word2Vec embedding](./image/word2vec.png)

### Glove
![Visualize Glove embedding](./image/glove.png)

### fastText
![Visualize fastText embedding](./image/fastText.png)


## Word embedding: Word2Vec, Glove, FastText
### A vector from Glove embedding (glove-wiki-gigaword-50)
```
King vector: [ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
  0.47377  -0.61798  -0.31012  -0.076666  1.493    -0.034189 -0.98173
  0.68229   0.81722  -0.51874  -0.31503  -0.55809   0.66421   0.1961
 -0.13495  -0.11476  -0.30344   0.41177  -2.223    -1.0756   -1.0783
 -0.34354   0.33505   1.9927   -0.04234  -0.64319   0.71125   0.49159
  0.16754   0.34344  -0.25663  -0.8523    0.1661    0.40102   1.1685
 -1.0137   -0.21585  -0.15155   0.78321  -0.91241  -1.6106   -0.64426
 -0.51042 ]
```

### Cosine similarity
```
King and Queen cosine similarity: 0.7839043
King and Man cosine similarity: 0.53093773
```

### Most similarity
```
Most similar words to 'computer':
computers: 0.9165045022964478
software: 0.8814992904663086
technology: 0.852556049823761
electronic: 0.812586784362793
internet: 0.8060455322265625
computing: 0.802603542804718
devices: 0.8016185760498047
digital: 0.7991793751716614
applications: 0.7912740707397461
pc: 0.7883159518241882
```

### Embed document
```
Document: "The queen rules the country."

Document vector: [-0.02883     0.38843602 -0.589208    0.0238326   0.04681259  0.19637859
 -0.304098   -0.114218   -0.01224605 -0.46948594  0.16859959  0.23931997
 -0.211428   -0.06647549  0.4421401   0.3574676  -0.00364    -0.05688141
 -0.211428   -0.06647549  0.4421401   0.3574676  -0.00364    -0.05688141
 -0.22242197 -0.22777598  0.205385    0.0287372  -0.0434166   0.09787501
 -0.04514321 -1.7586     -0.49889272 -0.093064   -0.127166    0.0500692
  3.2685401   0.14209768 -0.45647603 -0.19275999  0.02858421 -0.05513442
 -0.22242197 -0.22777598  0.205385    0.0287372  -0.0434166   0.09787501
 -0.22242197 -0.22777598  0.205385    0.0287372  -0.0434166   0.09787501
 -0.04514321 -1.7586     -0.49889272 -0.093064   -0.127166    0.0500692
 -0.22242197 -0.22777598  0.205385    0.0287372  -0.0434166   0.09787501
 -0.22242197 -0.22777598  0.205385    0.0287372  -0.0434166   0.09787501
 -0.04514321 -1.7586     -0.49889272 -0.093064   -0.127166    0.0500692
  3.2685401   0.14209768 -0.45647603 -0.19275999  0.02858421 -0.05513442
  0.212086   -0.1230404  -0.34473377 -0.28097197 -0.31398875 -0.01450661
  0.2132518  -0.00789342 -0.356534    0.22726226 -0.34674424 -0.39744362
  0.01643366 -0.40680504]
```

