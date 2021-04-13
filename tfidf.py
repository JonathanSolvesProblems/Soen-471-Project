from pyspark.sql import functions as func
from pyspark.sql import Row
from pyspark.sql.functions import lit, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.feature import Word2Vec, HashingTF, IDF, Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, FloatType
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, when

def display_word_count(df):
    # make new column in df of words, removing empty words and setting them all to lowercase
    words = df.select(func.explode(func.split(df.body, "\\W+")).alias("word"))
    words = words.filter(words.word != "")
    words = words.select(func.lower(words.word).alias("word"))

    # count the occurences of each word in descending order
    words.groupBy("word").count().sort("count", ascending = False).show()

    return words

def word2Vec(df):

    words = display_word_count(df)

    # word2vec model, counting the similarities of a word to others in vector space
    words = words.rdd.map(lambda x: x[0].split(","))
    model = Word2Vec().setVectorSize(10).setSeed(42).fit(words)
    similarities = model.findSynonyms('shares', 40)

    for word, dist in similarities:
        print("%s %f" % (word, dist))

def hashTF(df):
    words = display_word_count(df)

    words = words.rdd.map(lambda x: x[0].split(","))

    hashingTF = HashingTF()
    tf = hashingTF.transform(words)

    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)

    # print(tfidf.collect())

def score_body(df):
    tokenizer = Tokenizer(inputCol = "body", outputCol = "words")
    wordsData = tokenizer.transform(df)

    hashingTF = HashingTF(inputCol = "words", outputCol = "rawFeatures", numFeatures = 20)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol = "rawFeatures", outputCol = "features")
    idfModel = idf.fit(featurizedData)
    scaledData = idfModel.transform(featurizedData)

    # scaledData.select("features").show(1, False)
    # scaledData.select("rawFeatures").show(1, False)

    sum_ = udf(lambda x: float(x.values.sum()), DoubleType())
    scaledData = scaledData.withColumn("body significance", sum_("features"))

    body_significance = scaledData.select("body significance")
    # lit(0) is slower than row_number().over(Window.orderBy(monotonically_increasing_id()), using faster option despite warning, review.
    df = df.withColumn("temp", row_number().over(Window.orderBy(monotonically_increasing_id())))
    body_significance = body_significance.withColumn("temp", row_number().over(Window.orderBy(monotonically_increasing_id())))
    df = df.join(body_significance, on=["temp"]).drop("temp")
    
    return df

def score_hashtag(df):
    # throws error if we don't filter null.
    filter_null = df.filter(df.hashtags != "null")
    tokenizer = Tokenizer(inputCol = "hashtags", outputCol = "words")
    wordsData = tokenizer.transform(filter_null)

    hashingTF = HashingTF(inputCol = "words", outputCol = "rawFeatures", numFeatures = 20)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol = "rawFeatures", outputCol = "features")
    idfModel = idf.fit(featurizedData)
    scaledData = idfModel.transform(featurizedData)

    sum_ = udf(lambda x: float(x.values.sum()), DoubleType())
    scaledData = scaledData.withColumn("hashtag significance", sum_("features"))

    hashtag_significance = scaledData.select("hashtag significance")

    df = df.withColumn("temp2", row_number().over(Window.orderBy(monotonically_increasing_id())))
    hashtag_significance = hashtag_significance.withColumn("temp2", row_number().over(Window.orderBy(monotonically_increasing_id())))
    df = df.join(hashtag_significance, on=["temp2"]).drop("temp2")

    return df

    # TODO: Omit outliers when have all data distrubted.

def range_upvotes(df):
        df = df.withColumn("upvotes", when(col("upvotes").isNull(), 0).when((col("upvotes") >= 0) & (col("upvotes") <= 5000), 1).when((col("upvotes") > 5000) & (col("upvotes") <= 10000), 2)\
                            .when((col("upvotes") > 10000) & (col("upvotes") <= 15000), 3).when((col("upvotes") > 15000) & (col("upvotes") <= 20000), 4)\
                            .when((col("upvotes") > 20000) & (col("upvotes") <= 25000), 5).when((col("upvotes") > 30000) & (col("upvotes") <= 35000), 6)\
                            .when((col("upvotes") > 35000) & (col("upvotes") <= 40000), 7).when((col("upvotes") > 40000) & (col("upvotes") <= 45000), 8)\
                            .when((col("upvotes") > 45000) & (col("upvotes") <= 50000), 9).when((col("upvotes") > 50000) & (col("upvotes") <= 55000), 10)\
                            .when((col("upvotes") > 55000) & (col("upvotes") <= 60000), 11).when((col("upvotes") > 60000) & (col("upvotes") <= 65000), 12)\
                            .when((col("upvotes") > 65000), 13))

        # TESTING - To Remove
        # data_df = data_df.withColumn("Plays", data_df["Plays"].cast(IntegerType()))
        # df = df.withColumn("urls_domain_modified", when(col("urls_domain_modified") != "", col("urls_domain_modified")).otherwise("No link"))
        # df = df.withColumn("upvotes_2", when((col("upvotes") >= 0) & (col("upvotes") <= 100), 1).otherwise("Meow"))
        # df = df.withColumn("upvotes", df["upvotes"].cast(IntegerType()))

        return df