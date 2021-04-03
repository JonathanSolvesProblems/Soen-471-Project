from pyspark.sql import functions as func
from pyspark.mllib.feature import HashingTF, IDF, Word2Vec

def word2Vec(df):
    # splitting words line by line
    words = df.select(func.explode(func.split(df.body, "\\W+")).alias("word"))
    # removing blanks
    words = words.filter(words.word != "")
    # lower casing all words in order to not diffentiate capitalization
    words = words.select(func.lower(words.word).alias("word"))
    words.groupBy("word").count().sort("count", ascending = False).show()

    words = words.rdd.map(lambda x: x[0])

    
    hashingTF = HashingTF()
    tf = hashingTF.transform(words)

    # computing IDF vector and scaling term frequencies by the IDF
    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)
    print(tfidf.collect()[0])