from preprocess import *
#from plot import *
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from tfidf import *
from pyspark.sql.functions import col, when
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

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
df = preprocessor.getProcessedData()
df = df.drop('body', 'createdAtformatted')

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
#categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []
# for categoricalCol in categoricalColumns:
#     stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
#     encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
#     stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'upvotes', outputCol = 'label')
stages = [label_stringIdx]
assembler = VectorAssembler(
    inputCols=['comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType', 'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance'],
    outputCol="features")
stages += [assembler]

output = assembler.transform(df)
output.select("features", "upvotes").show(truncate=False)
output.show()

from pyspark.ml import Pipeline
cols = df.columns
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()

train, val, test = df.randomSplit([0.6, 0.2, 0.2])

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = df.randomSplit([0.7, 0.3])
# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")
# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])
# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)
# Make predictions.
predictions = model.transform(testData)
# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)



#print(train.count())
#print(test.count())
#df.show()

# train_x = train.select('comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType', 'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance')
# train_y = train.select('upvotes')
# val_x = val.select('comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType', 'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance')
# val_y = val.select('upvotes')
# test_x = test.select('comments', 'followers', 'following', 'impressions', 'reposts', 'verified', 'categoryIndexMimeType', 'categoryIndexDomains', 'sentiment_score', 'hashtag significance', 'body significance')
# test_y = test.select('upvotes')

# train.show()
# # #BUILD MODEL
# from pyspark.ml import Pipeline
#
# featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(output)
# (trainingData, testData) = output.randomSplit([0.7, 0.3])
#
# # Train a RandomForest model.
# rf = RandomForestRegressor(featuresCol="indexedFeatures")
#
# # Chain indexer and forest in a Pipeline
# pipeline = Pipeline(stages=[featureIndexer, rf])
#
# # Train model.  This also runs the indexer.
# model = pipeline.fit(trainingData)
#
# # Make predictions.
# predictions = model.transform(testData)

# rf = RandomForestRegressor(featuresCol = 'features')
# rfModel = rf.fit(train)
# predictions = rfModel.transform(test)
# #test_y.show()
# predictions.select('upvotes').show()