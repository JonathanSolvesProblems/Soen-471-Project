from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
# from pyspark.sql.functions import lit
# from pyspark.sql.functions import desc
from pyspark.sql.types import *
from pyspark.sql.functions import *
import sys
import os
from pathlib import Path
from pyspark.sql.functions import col, when


class Preprocess(object):
    def __init__(self, inputJsonDirectory, outputFileDirectory, outputJson):
        self.inputJsonDirectory = inputJsonDirectory
        self.outputFileDirectory = outputFileDirectory  
        self.outputJson = outputJson
        self.spark = self.init_spark()

    def init_spark(self):
        
        spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        #conf = SparkConf().setAppName("test").setMaster("local")
        #sc = SparkContext(conf=conf)
        return spark

    def getSpark(self):
        return self.spark

    def preprocessJson(self, inputJsonDirectory):

        print(inputJsonDirectory)
        df = self.spark.read.json(inputJsonDirectory)
        print(df.count())

        # df.show()
        
        df = self.flatten(df)

        # Here we start dropping columns
        
        # remove -1 comments
        df = df.filter(df.comments != -1)

        # dropping columns
        drop_columns = ['username', 'bodywithurls', 'depth', 'depthRaw', 'lastseents', 'links', 'media', 'parent', 'posts', 'preview', 'state', 'urls_createdAt', 'urls_id', 'urls_modified',
        'urls_short', 'urls_state', 'createdAt', 'urls_long', 'urls_metadata_length']

        df = df.drop(*drop_columns)


        df = df.withColumn("urls_domain", lambda x : when(col(x) != "", col(x)).otherwise("No link"))
        
        
        # df.write.format("csv").save(self.outputJson)

        
        
        # paralizing later, just for testing
        df.toPandas().to_csv(self.outputJson)

        # df.printSchema()


        # testing
        # df.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("mydata.csv")
        
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

    def createResultDirectory(self):
        # TODO: add outputFileDirectory, was getting weird error with it
        # output_path = self.outputFileDirectory + self.outputJson
        try:
            f = open(self.outputJson, "w")
            f.write("TODO: Add Results to CSV")
            f.close()
        except:
            sys.exit("Error: Unable to create file.")