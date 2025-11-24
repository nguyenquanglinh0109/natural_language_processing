# **Lab 3: Word Embeddings**
## Word embedding
### Implement
```
1. Implement WordEmbedder use glove-wiki-gigaword-50 model with methods:
  + get vector: get vector of a word
  + get similar: calculate the similar between two words
  + get most similarity: get the most (top-n) similarity with a word
  + embed document: tokenizer, tranform tokens to vectors and mean.

2. Word2Vec with spark
```

### Run code
```
py -m test.lab4_test
py -m test.lab4_embedding_traninng_demo
py -m test.lab4_spark_word2vec_demo
```

### Result
#### A vector from Glove embedding (glove-wiki-gigaword-50)
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

#### Cosine similarity
```
King and Queen cosine similarity: 0.7839043
King and Man cosine similarity: 0.53093773
```

#### Most similarity
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

#### Embed document
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

#### Similarity from trained model Word2Vec
```
+---------+------------------+
|     word|        similarity|
+---------+------------------+
|  desktop|0.6893201470375061|
|computers|0.6467169523239136|
|   laptop|0.6318738460540771|
| software|0.5909299254417419|
|   device|0.5804780721664429|
+---------+------------------+
```


#### Explain
```
+ Pretrained embedding lấy từ model glove-wiki-gigaword-50, mỗi từ được ánh xạ sang một vector gồm 50 chiều, như trên từ 'king' được biểu diễn bởi một vector 50 chiều.
+ Cosine similar: đo độ tương đồng giữa hai vector từ bằng công thức cosine, giá trị càng gần 1 thì độ tương đồng càng lớn.
+ Most similarity: dựa vào cosine similar giữa từng cặp từ để trả về top_k từ có độ tương đồng lớn nhất, ở đây danh sách các từ gần với 'computer' trong không gian vector đều có sự tương đồng và liên quan lớn tới từ này trong thực tế.
+ Embed document: document được tách thành các token với RegexTokenizer, sau đó từng token sẽ được chuyển thành vector dựa trên pretrained model rồi lấy trung bình của toàn bộ vector khi đó ta được một vector biểu diễn đại diện cho toàn bộ câu.
+ Similarity from trained model Word2Vec: cho kết quả tương đối tốt, khi các từ trong top-5 (desktop, computers, laptop, software, device) đều gần nghĩa hoặc có liên quan tới từ 'computer'. Tuy nhiên so với điểm số similarity từ pretrained model có thể thấy được khả năng biểu diễn ngữ nghĩa chưa mạnh mẽ bằng.
```


## Visualize work embedding with PCA
### Implement
```
Get pretrained embedding: "word2vec-google-news-300", "glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300".
Visualize with PCA.
```

### Run code
```
word2vec_vector = gensim.downloader.load("word2vec-google-news-300")
glove_vector = gensim.downloader.load("glove-wiki-gigaword-300")
fasttext_vector = gensim.downloader.load("fasttext-wiki-news-subwords-300")

Run file 'word_embedding_visualization.ipynb'
```

### Result
#### Word2Vec
![Visualize Word2Vec embedding](./../image/word2vec.png)

#### Glove
![Visualize Glove embedding](./../image/glove.png)

#### fastText
![Visualize fastText embedding](./../image/fastText.png)

#### Explain
```
Trong phần trực quan hoá PCA với Word2Vec có thể thấy được mối quan hệ tuyến tính giữa các cặp từ: king-queen, man-woman, boy-girl; các đường nối giữa chúng dường như gần song song với nhau cho thấy khả năng biểu diễn ngữ nghĩa của Word2Vec do phương pháp huấn luyện những cặp từ này thường xuất hiện cùng nhau.

Với Glove kết quả cho ra giữa các cặp thủ đô-đất nước cũng cho kết quả tương đối giống Word2Vec.

Với fastText, kết quả cho ra không thể hiện rõ mối quan hệ tuyến tính giữa các cặp từ; tuy nhiên với các cặp từ gần nghĩa hoặc thường xuất hiện trong cùng ngữ cảnh thì tạo thành cụm tương đối rõ rêt: cụm boy-girl, man-woman, king-queen
```

## Problem:
1. Khi huấn luyện mô hình Word2Vec bị tràn bộ nhớ -> lọc bỏ bớt các từ xuất hiện ít hơn 2 lần.


## Reference: 
[Gensim](https://radimrehurek.com/gensim/) \
[PySpark](https://spark.apache.org/docs/latest/api/python/reference/index.html)

## Pretrain models
```
"word2vec-google-news-300", 
"glove-wiki-gigaword-300", 
"fasttext-wiki-news-subwords-300"
```
