from pyspark.sql import functions as func
from pyspark.sql import Row
from pyspark.ml.feature import Word2Vec, HashingTF, IDF, Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf

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

    scaledData.select("body significance").show()
    
    return scaledData

def score_hashtag(df):
    tokenizer = Tokenizer(inputCol = "hashtags", outputCol = "words")
    wordsData = tokenizer.transform(df)

    hashingTF = HashingTF(inputCol = "words", outputCol = "rawFeatures", numFeatures = 20)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol = "rawFeatures", outputCol = "features")
    idfModel = idf.fit(featurizedData)
    scaledData = idfModel.transform(featurizedData)

    # sum_ = udf(lambda x: float(x.values.sum()), DoubleType())
    # scaledData = scaledData.withColumn("hashtag significance", sum_("features"))

    # scaledData.select("hashtag significance").show()
    
    # return scaledData

    # TODO: Add sentiment analysis
    # TODO: Do same thing on hashtags
    # TODO: Double check stop words.
    # TODO: Categorizing
    # TODO: Use regex to convert media0.giphy into one
    # TODO: Use vador and check how polarize the opinion is.