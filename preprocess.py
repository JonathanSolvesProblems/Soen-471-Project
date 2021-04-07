from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
# from pyspark.sql.functions import lit
# from pyspark.sql.functions import desc
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
import sys
import os
from pathlib import Path
from pyspark.sql.functions import col, when
import matplotlib.pyplot as plt
import shutil
from tfidf import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from textblob import TextBlob
from pyspark.sql.functions import regexp_replace


class Preprocess(object):
    def __init__(self, inputJsonDirectory, outputFileDirectory, outputJson):
        self.inputJsonDirectory = inputJsonDirectory
        self.outputFileDirectory = outputFileDirectory  
        self.outputJson = outputJson
        self.spark = self.init_spark()
        self.dfPre = None
        self.dfPost = None

    def init_spark(self):
        
        spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        #conf = SparkConf().setAppName("test").setMaster("local")
        #sc = SparkContext(conf=conf)
        return spark

    def getPreprocessedData(self):
        return self.dfPre

    def getProcessedData(self):
        return self.dfPost

    def getSpark(self):
        return self.spark


    def preprocessJson(self, inputJsonDirectory):

        # print(inputJsonDirectory)
        df = self.spark.read.json(inputJsonDirectory)
        # print(df.count())

        # df.show()
        
        df = self.flatten(df)
        self.dfPre = df

        # Here we start dropping columns
        # remove -1 comments
        df = df.filter(df.comments != -1)

        # dropping columns
        drop_columns = ['username', 'bodywithurls', 'depth', 'depthRaw', 'lastseents', 'links', 'media', 'parent', 'posts', 'preview', 'state', 'urls_createdAt', 'urls_id', 'urls_modified',
        'urls_short', 'urls_state', 'createdAt', 'urls_long', 'urls_metadata_length']

        df = df.drop(*drop_columns)
        # df = self.resample(df, 0.5, 'upvotes', 100) # <- something off with this function, review.
        

        df = df.withColumn("urls_domain_modified", regexp_replace(col("urls_domain"), "media\d*.giphy.com", "media0.giphy.com"))
        df = df.withColumn("urls_metadata_mimeType", when(col("urls_metadata_mimeType") != "", col("urls_metadata_mimeType")).otherwise("No mimetype"))

        df = df.withColumn("urls_domain_modified", when(col("urls_domain_modified") != "", col("urls_domain_modified")).otherwise("No link"))
        indexer = StringIndexer(inputCol="urls_metadata_mimeType", outputCol="categoryIndexMimeType")
        indexed = indexer.fit(df).transform(df)
        df = indexed
        
        indexer = StringIndexer(inputCol="urls_domain_modified", outputCol="categoryIndexDomains")
        indexed = indexer.fit(df).transform(df)
        df = indexed
        df = df.withColumn('article', F.when(df.verified == 'FALSE', 0).otherwise(1))
        df = df.withColumn('sensitive', F.when(df.verified == 'FALSE', 0).otherwise(1))
        df = df.withColumn('verified', F.when(df.verified == 'FALSE', 0).otherwise(1))
        df = df.filter((df.article == '0') | (df.article == '1'))
        # indexed.show()

        def sentiment_analysis(text):
            return TextBlob(text).sentiment.polarity
        sentiment_analysis_udf = udf(sentiment_analysis , FloatType())
        df = df.withColumn("sentiment_score", sentiment_analysis_udf( df['body']))

        
        
        dirpath = Path(self.outputJson)
        # Checking if path exists, if true delete everything inside folder
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)

        df = score_hashtag(df)
        
        new_drop = ["urls_domain_modified", "article", "hashtags", "urls_metadata_site", "urls_metadata_mimeType", "score", "urls_domain", "shareLink", "color", "commentDepth", "controversy", "conversation", "creator", "datatype", "downvotes", "id", "isPrimary", "post", "replyingTo"]
        df = df.drop(*new_drop)

        df = df.na.drop(subset = ["impressions", "reposts"])
        df = df.dropDuplicates()

        df = score_body(df)

        self.dfPost = df

        df.coalesce(1).write.format("csv").save(self.outputJson, header = True)

        # df.show(1)

        return 0

    def flatten(self, df):
        # compute Complex Fields (Lists and Structs) in Schema   
        complex_fields = dict([(field.name, field.dataType)
                                 for field in df.schema.fields
                                 if type(field.dataType) == ArrayType or  type(field.dataType) == StructType])
        while len(complex_fields)!=0:
          col_name=list(complex_fields.keys())[0]
          print ("Processing :"+col_name+" Type : "+str(type(complex_fields[col_name])))

          # if StructType then convert all sub element to columns.
          # i.e. flatten structs
          if (type(complex_fields[col_name]) == StructType):
             expanded = [col(col_name+'.'+k).alias(col_name+'_'+k) for k in [ n.name for n in  complex_fields[col_name]]]
             df=df.select("*", *expanded).drop(col_name)

          # if ArrayType then add the Array Elements as Rows using the explode function
          # i.e. explode Arrays
          elif (type(complex_fields[col_name]) == ArrayType):    
             df=df.withColumn(col_name,explode_outer(col_name))

          # recompute remaining Complex Fields in Schema       
          complex_fields = dict([(field.name, field.dataType)
                                 for field in df.schema.fields
                                 if type(field.dataType) == ArrayType or  type(field.dataType) == StructType])
        return df


    def resample(self, base_features,ratio,class_field,base_class):
        pos = base_features.filter(col(class_field)>=base_class)
        neg = base_features.filter(col(class_field)<base_class)
        total_pos = pos.count()
        total_neg = neg.count()
        fraction=float(total_pos*ratio)/float(total_neg)
        sampled = neg.sample(False,fraction)
        return sampled.union(pos)

    def reprocessed_data(self):
        # TO REMOVE, was for testing
        df = self.spark.read.csv("./parler-data/*.csv", header = True)
        # df.show(1)

