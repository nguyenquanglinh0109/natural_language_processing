from webbrowser import get
import pip
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    RegexTokenizer, 
    CountVectorizer, 
    StringIndexer
)
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pandas as pd
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.sql import DataFrame
spark = SparkSession.builder.appName("TextClassification").getOrCreate()

def get_data(data_path: str) -> DataFrame:
    """Get data from csv file

    Args:
        data_path (str): _path of csv file

    Returns:
        DataFrame: _pyspark dataframe
    """
    df = spark.read.option("header", "true").csv(data_path)
    df = df.dropna()
    return df

def get_pipeline(indexer: StringIndexer, tokenizer: RegexTokenizer, vectorizer: CountVectorizer) -> Pipeline:
    return Pipeline(stages=[indexer, tokenizer, vectorizer])

def get_model(pipeline: Pipeline, model) -> Pipeline:
    pipeline.addStages(model)
    return pipeline

def main():
    df = get_data(r".\data\sentiments.csv")
    indexer = StringIndexer(inputCol="sentiment", outputCol="label")
    tokenizer = RegexTokenizer(inputCol="text", outputCol="words")
    vectorizer = CountVectorizer(inputCol="words", outputCol="features")
    pipeline = get_pipeline(indexer, tokenizer, vectorizer)
    model = pipeline.fit(df)
    df = model.transform(df)
    df.show()

if __name__ == "__main__":
    ## 1. get feature from data
    main()

    ## 2. logistic regression
    # df = get_data("../data/sentiment.csv")
    # train, test = df.randomSplit([0.8, 0.2])

    # indexer = StringIndexer(inputCol="sentiment", outputCol="label")
    # tokenizer = RegexTokenizer(inputCol="text", outputCol="words")
    # vectorizer = CountVectorizer(inputCol="words", outputCol="features")
    # pipeline = get_pipeline(indexer, tokenizer, vectorizer)
    # model = pipeline.fit(df)

    # lr = LogisticRegression(featuresCol="features", labelCol="label")
    # lr_pipeline = get_model(pipeline, lr)
    # lr_model = lr_pipeline.fit(train)
    # lr_results = lr_model.transform(test)
    # lr_evaluator = BinaryClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
    # print(lr_evaluator.evaluate(lr_results))


    ## 3. multilayer perceptron
    # mlp = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label")
    # mlp_pipeline = get_model(pipeline, mlp)
    # mlp_model = mlp_pipeline.fit(train)
    # mlp_results = mlp_model.transform(test)
    # mlp_evaluator = BinaryClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
    # print(mlp_evaluator.evaluate(mlp_results))