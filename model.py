from preprocess import *
#from plot import *
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from tfidf import *
from pyspark.sql.functions import col, when


parlerDataDirectory = './parler_small.ndjson/'
outputFileDirectory = './preprocessed/'
outputJson = './parlers-data/'
upvoteBins = 4

preprocessor = Preprocess(parlerDataDirectory, outputFileDirectory, outputJson)
preprocessor.preprocessJson(parlerDataDirectory)
# plotUpvotes(preprocessor.getPreprocessedData(), upvoteBins)
# plotUpvotes(preprocessor.getProcessedData(), upvoteBins)

# categoricalColumns = ['article', 'body', 'createdAtformatted', 'creator', 'datatype', 'hashtags', 'id', 'sensitive', 'shareLink', 'verified', 'urls_domain', 'urls_metadata_mimeType', 'urls_metadata_site']
# stages = []
# for categoricalCol in categoricalColumns:
#     stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
#     encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
#     stages += [stringIndexer, encoder]
# label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
# stages += [label_stringIdx]
# numericCols = ['comments', 'followers', 'following', 'impressions', 'reposts']
# assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
# assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
# stages += [assembler]


#PREP DATA
df = preprocessor.getProcessedData()
body_vector = score_body(df)
#hf = score_hashtag(df)
#df = body_vector.join(df, on=["row_index"]).drop("row_index")
print(type(body_vector['body significance']))
import pyspark.sql.functions as F

body_vector = body_vector.select(F.col("body significance").alias("body significance"))

df = df.withColumn('body significance', body_vector['body significance'])
df.show()
# cols = df.columns
#
# from pyspark.ml import Pipeline
# pipeline = Pipeline(stages = stages)
# pipelineModel = pipeline.fit(df)
# df = pipelineModel.transform(df)
# selectedCols = ['label', 'features'] + cols
# df = df.select(selectedCols)
# df.printSchema()


train, val, test = df.randomSplit([0.6, 0.2, 0.2])
#print(train.count())
#print(test.count())
#df.show()

# train_x = train.select('body', 'comments', 'createdAtformatted', 'creator', 'followers', 'following', 'hashtags', 'id', 'impressions', 'reposts', 'verified')
# train_y = train.select('upvotes')
# val_x = val.select('body', 'comments', 'createdAtformatted', 'creator', 'followers', 'following', 'hashtags', 'id', 'impressions', 'reposts', 'verified')
# val_y = val.select('upvotes')
# test_x = test.select('body', 'comments', 'createdAtformatted', 'creator', 'followers', 'following', 'hashtags', 'id', 'impressions', 'reposts', 'verified')
# test_y = test.select('upvotes')
#
# #BUILD MODEL
# rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
# rfModel = rf.fit(train)
# predictions = rfModel.transform(test)
# #test_y.show()
# predictions.select('upvotes').show()