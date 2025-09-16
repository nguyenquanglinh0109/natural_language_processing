# **Week 1**

---

# **Week 2**

## Implement SimpleTokenizer and RegexTokenizer

```text
Hello, world! This is a test.

['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
==================================================

NLP is fascinating... isn't it?

['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
['NLP', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']
==================================================

Let's see how it handles 123 numbers and punctuation!

["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
['Let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
==================================================

## Implement Bag of Words
### Corpus
['I love NLP, you love programming.', 'I love programming.', 'NLP is a subfield of AI.']

### Vocabulary
{'<UNK>': 0, ',': 1, '.': 2, 'AI': 3, 'I': 4, 'NLP': 5, 'a': 6, 'is': 7, 'love': 8, 'of': 9, 'programming': 10, 'subfield': 11, 'you': 12}

### Bag of Words representation
[[0, 1, 1, 0, 1, 1, 0, 0, 2, 0, 1, 0, 1], 
 [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], 
 [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]]

