from pyspark.sql import functions as func
from pyspark.sql import Row
from pyspark.ml.feature import Word2Vec, HashingTF, IDF, Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

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

def score(df):
    tokenizer = Tokenizer(inputCol = "body", outputCol = "words")
    wordsData = tokenizer.transform(df)

    hashingTF = HashingTF(inputCol = "words", outputCol = "rawFeatures", numFeatures = 20)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol = "rawFeatures", outputCol = "features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData.select("features").show(1, False)

    print(hashingTF)
    rescaledData.printSchema()
    rescaledData.select("rawFeatures").show(1, False)

    # res = rescaledData.rdd.map(lambda x : (x.features[0][2],(None if x.features is None else x.features.sum())))

    from pyspark.sql.functions import udf

    sum_ = udf(lambda x: float(x.values.sum()), DoubleType())
    rescaledData = rescaledData.withColumn("idf_sum", sum_("features"))

    rescaledData.select("idf_sum").show()

    # TODO: Add sentiment analysis
    # TODO: Do same thing on hashtags
    # TODO: Double check stop words.
    # TODO: Categorizing
    # TODO: Use regex to convert media0.giphy into one
    # TODO: Use vador and check how polarize the opinion is.
    
    return rescaledData

    #significance = rescaledData.select(func.sum(rescaledData.features).alias("significance"))

    #significance.select("significance").show()

    # df = rescaledData.select("body", "features")

### BACKUP, to remove, failed attempts:
# stopwords.words('english')
    # separate words by sentences
    # df = df.filter(df.body != "")
    #sentences = df.select("body")
    #sentences = sentences.rdd.map(lambda x : x[0])
    #sentences = sentences.map(lambda x : x.split(" "))

    #sentences.show()

    # bag_of_words = df.select(func.explode(func.split(df.body, "\\W+")).alias("word"))
    # unique_bag_of_words = bag_of_words.select("word").dropDuplicates()
    # unique_bag_of_words = unique_bag_of_words.rdd.map(lambda x : x[0])

    
    # df = df.select("body").rdd.map(lambda x : x.body.split(" ")).toDF().withColumnRenamed("_1","body")
    
    # # df.show()
    
    # htf = HashingTF(inputCol="body", outputCol="tf")
    # tf = htf.transform(df)
    # tf.show(truncate=False)


    #sum_ = udf(lambda v: float(v.sentences.sum()), DoubleType())
    #tfidf.withColumn("idf_sum", sum_("idf")).show()

    # filtered_data = 

    # vectorizer = TfidfVectorizer()

    # for i in filtered_data:
    #     vectors = vectorizer.fit_transform([docA, docB])

    # print(unique_bag_of_words.collect())