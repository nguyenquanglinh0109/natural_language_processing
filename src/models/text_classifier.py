"""
1.masking
2. embedding
3. lstm
4. dense 
loss: cross entropy loss
tensorboard
"""
from src.core.interfaces import Vectorizer
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

class TextClassifier:
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: List[str], labels: List[int]):
        """Fit model classifier

        Args:
            texts (List[str]): input texts
            labels (List[int]): label

        Returns:
            _type_: model classifier
        """
        print("fitting model...")
        self._model = LogisticRegression(solver="liblinear")
        vectors = self.vectorizer.fit_transform(texts)
        self._model.fit(vectors, labels)
        
        return self._model
    
    def predict(self, texts: List[str]) -> List[int]:
        """Predict the label

        Args:
            texts (List[str]): the input

        Raises:
            ValueError: error when model didn't train

        Returns:
            List[int]: outputs
        """
        if self._model is None:
            raise ValueError("Don't have model, use fit() to create")
  
        vectors = self.vectorizer.transform(texts)
        outputs = self._model.predict(vectors)
        
        return outputs
    
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str,
float]: 
        """
        Args:
            y_true (List[int]): true labels
            y_pred (List[int]): predicted labels

        Returns:
            Dict[str, float]: metrics
        """ 
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        results = {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        return results