from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    RegexTokenizer, 
    CountVectorizer, 
    StringIndexer,
    StopWordsRemover,
    HashingTF,
    IDF
)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

def init_spark():
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    return spark

def get_data(data_path: str, spark: SparkSession) -> DataFrame:
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    # Convert -1/1 labels to 0/1: Normalize sentiment labels
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    # Drop rows with null sentiment values before processing
    initial_row_count = df.count()
    df = df.dropna(subset=["sentiment"])
    return df
    
def get_preprocessing_pipeline():
    tokenizer = RegexTokenizer(inputCol="text", outputCol="words")
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features")
    idf = IDF(inputCol="raw_features", outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf])
    return pipeline

def train_model(training_data: DataFrame):
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
    pipeline = get_preprocessing_pipeline()
    pipeline.getStages().append(lr)
    model = pipeline.fit(training_data)
    return model

def evaluate_model(model, test_data: DataFrame):
    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    return accuracy, precision,recall, f1



def main():
    spark = init_spark()
    print(spark.sparkContext.uiWebUrl)
    df = get_data("data/sentiments.csv", spark)
    train, test = df.randomSplit([0.8, 0.2])
    model = train_model(train)
    accuracy, precision, recall, f1 = evaluate_model(model, test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    input("Press Enter to continue...")
    spark.stop()

if __name__ == "__main__":
    main()
    
    