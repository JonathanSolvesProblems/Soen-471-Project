from pyspark.sql import functions as func
from pyspark.mllib.feature import Word2Vec, HashingTF, IDF

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

    print(tfidf.collect())

