import re
import json 

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.sql import DataFrame

def init_spark():
    spark = SparkSession.builder.appName("Word2VecDemo").getOrCreate()
    return spark

def load_data(data_path: str, spark: SparkSession) -> DataFrame:
    df = spark.read.json(data_path)
    
    return df

def preprocessing(df: DataFrame) -> DataFrame:
    df = df.withColumn("text", lower(col("text")))
    df = df.withColumn("text", regexp_replace(col("text"), r"[^\w\s]", ""))
    df = df.withColumn("words", split(col("text"), " "))
    return df

def train_model(df: DataFrame) -> Word2Vec:
    w2v_model = Word2Vec(
        vectorSize=100,
        maxIter=10,
        minCount=2,
        windowSize=2,
        seed=42,
        inputCol="words",
        outputCol="word_vector"
    )
    
    model = w2v_model.fit(df)
    return model
    
    
def main():
    # Initialize Spark Session
    print("Initializing Spark Session...")
    spark = init_spark()
    
    # Load the dataset
    print("Loading dataset...")
    data_path = "data/c4-train.00000-of-01024-30K.json/c4-train.00000-of-01024-30K.json"
    df = load_data(data_path, spark)
    print(df.show(5))
    
    # Preprocessing
    print("Preprocessing...")
    df = preprocessing(df)
    print(df.show(5))
    
    # Configure and train the Word2Vec model
    print("Training model...")
    model = train_model(df)
    
    # Find synonyms for a word
    vocab = model.getVectors()
    if vocab.filter(col("word") == "computer").count() == 0:
        first_word = vocab.select("word").first()[0]
        print(f"'computers' not in vocab. Trying with '{first_word}' instead.")
        synonyms = model.findSynonyms(first_word, 5)
    else:
        synonyms = model.findSynonyms("computer", 5)

    synonyms.show(5)
    
    # Stop the Spark session
    input("Press any key to stop...")
    spark.stop()
    
    
if __name__ == "__main__":
    main()