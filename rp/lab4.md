# **Lab 4: Text classification**#
## Implement
```
1. Implement text_classifier.py with some methods:
  + fit: train logistic regression model from sklearn
  + predict: use the lr model to predict test dataset
  + evaluate: return acc, precision, recall, f1_score

2. Implement spark_sentiment_analysis.py:
  + preprocessing: Tokenizer, StopWordsRemover, HashingTF, IDF
  + train model: Logistic Regression
  + evaluate: return accuracy

3. Improve text classifier: improvement_test.py
  + preprocessing: remove uncommon word
  + model: change lr -> gradient boost
```

## Run code
```
py -m test.lab5_test
py -m test.lab5_spark_sentiment_analysis
py -m test.lab5_improvement_test
```

## Result
### Simple classifier
```
data_path = "data\sentiments.csv"
Run: py -m test.lab5_test

Predicts: [ 1  1  1  1 -1  1 -1 -1 -1  1  1  1 -1  1  1 -1  1  1  1 -1 ...]
True: [-1 -1  1 -1 -1  1  1 -1 -1  1  1  1 -1 -1  1 -1  1  1  1 -1 ...]

Evaluation:
  + Accuracy: 0.7947
  + Precision: 0.8183
  + Recall: 0.8675
  + F1: 0.8422
```

### Spark sentiment analyis
```
Run: py -m test.lab5_spark_sentiment_analysis

Evaluation:
  + Accuracy: 0.7333
  + Precision: 0.7314
  + Recall: 0.7333
  + F1-score: 0.7322
```

### Improvement
```
Run: py -m test.lab5_improvement_test

Uncommon words: ['sooner', 'indicating', 'phones', '166', 'fav', 'reminder', 'reporting', 'patent', 'deals', 'cien', 'th', 'dds', 'smaller', 'email', 'chain', 'commentary', 'initiated', 'msh', '2006', '470', 'charm', 'te', 'whisper', 'tweet', 'skx', 'formed', 'uncertainty', 'pennant', 'raising', 'website', 'type', 'earning', 'model', 'strain', 'epidemic', 'peaks', 'beâ', 'paid', 'kirby', 'toe', 'search', 'tes', 'collapse', 'speculation', 'included', 'wnc', 'nti', 'ife', 'daytrade', 'nailed', 'quietly', 'send', 'fas', 'gdi', 'onxx', 'forces', 'cbs', 'nxt', 'smartphone', 'itâ', 'emerging', 'bcd', 'advisors', 'purchases', 'doubled', 'employee', 'bases', 'thank', 'ty', 'ceiling', '434', 'dude', 'amwd', 'mp', ...]

Evaluation:
  + Accuracy: 0.7023
  + Precision: 0.7065
  + Recall: 0.9044
  + F1: 0.7933

```

## Explain
```
Dữ liệu được lấy từ: "data\sentiment.csv", chia thành train và test với tỉ lệ 80:20.
Toàn bộ sau đó câu sau đó được tokenizer, sau đó được vector hoá với 2 phương pháp: CountVectorizer, Pretrained embedding
Các chỉ số đánh giá được lấy từ kết quả dự đoán của mô hình so với nhãn thât.

Phương pháp gốc: CountVectorizer + Logistic Regression cho kết quả tương đối tốt với Acc: 79,47% và F1: 84,22%

Spark: RegexTokenizer + StopwordRemover + HashingTF + IDF, cho kết quả thấp hơn với Acc: 73,33 và F1: 73,22

Phương pháp cải thiện: RegexTokenizer + RemoveUncommonWord + GradientBoostingClassifier, cho kết quả với Acc: 0.7023 và F1: 0.7933
```

## Problem
1. Đã thử cải thiện bằng một số phương pháp tiền xử lý, cũng như thay đổi thành các mô hình phân loại hiệu quả hơn tuy nhiên các phương pháp đều cho hiệu suất kém hơn so với mô hình gốc ban đầu.

## Reference
[Sklearn](https://scikit-learn.org/stable/index.html) \
[PySpark](https://spark.apache.org/docs/latest/api/python/reference/index.html)

