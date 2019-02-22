# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Cluster Analysis
# MAGIC <br>
# MAGIC ## Task
# MAGIC 
# MAGIC * Group the facebook posts into groups so that the groups contain posts with similar content.
# MAGIC * This task has no unique solution - the data is not labled so we don't know the correct answer therefore you will not be able to verify how good your model is
# MAGIC 
# MAGIC ## Data
# MAGIC * We prepared two datasets about facebook pages
# MAGIC * The first dataset contains list of pages
# MAGIC * The second dataset contains for each page 100 randomly selected posts
# MAGIC * You will actualy not need to use the pages set unless you come up with some interesting features that will be useful
# MAGIC 
# MAGIC ## Notes
# MAGIC * Create vector reprezentation for the texts
# MAGIC * Split the text to words
# MAGIC * Compute IDF
# MAGIC * Use KMeans algorithm to train a model
# MAGIC 
# MAGIC ## About K-Means
# MAGIC * K-means is a unsupervised learning algorithm that can be used to cluster data into groups
# MAGIC * For details see <a target="_blank" href="https://en.wikipedia.org/wiki/K-means_clustering">wiki</a>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Documentation
# MAGIC <br>
# MAGIC * Pyspark documentation of DataFrame API is <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html">here</a>
# MAGIC 
# MAGIC * Pyspark documentation of ML Pipelines library is <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html">here</a>
# MAGIC 
# MAGIC * Prezentation slides are accessed <a target="_blank" href = "https://docs.google.com/presentation/d/1XNKIfE5Atj_Mzse0wjmbwLecmVs2YkWm9cqOLqDVWPo/edit?usp=sharing">here</a> 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Import functions

# COMMAND ----------

from pyspark.sql.functions import col, count, desc, row_number, collect_list, length, array_contains, size
from pyspark.sql.functions import col, count, desc, array_contains, broadcast, explode, length, first, when

from pyspark.sql import Window

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer, SQLTransformer

from pyspark.ml.clustering import KMeans

from pyspark.ml import Pipeline

import random

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Load the data

# COMMAND ----------

pages = spark.table('mlprague.facebook_pages')

posts = spark.table('mlprague.facebook_posts')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### You may want to do some exploratory analytics first
# MAGIC 
# MAGIC hint:
# MAGIC * see how many records you have
# MAGIC * what is the schema of the dataset
# MAGIC * see some records
# MAGIC * use can use printSchema(), show(), count(), or proprietaray function display()

# COMMAND ----------

pages.count()

# COMMAND ----------

posts.count()

# COMMAND ----------

display(pages)

# COMMAND ----------

display(posts)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Extract the features & construct the pipeline
# MAGIC 
# MAGIC hint
# MAGIC * do vector representation for the texts
# MAGIC  * use: 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Tokenizer">Tokenizer</a> 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StopWordsRemover">StopWordsRemover</a> 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.HashingTF">HashingTF</a> to compute term frequency and reduce the space or use the <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.CountVectorizer">CountVectorizer</a>
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.IDF">IDF</a> 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Normalizer">Normalizer</a> 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.KMeans">KMeans</a> 
# MAGIC * See the slides 83, 84, 85, 101 in the presentation
# MAGIC * after you apply StopWordsRemover it is good to filter out rows with no (or very few) words. You can use the SQLTransformer defined bellow, that filters out all rows that have less than 10 words. This transformer assumes that the output column of StopWordsRemover is named 'noStopWords'. Just add this SQLTransformer to the pipeline right behind the StopwordsRemover
# MAGIC 
# MAGIC Note
# MAGIC * You may want to play with some input parameters: 
# MAGIC  * number of clusters for KMeans (try 4-8)
# MAGIC  * distanceMeasure for KMeans (default is 'euclidean' but you can try also 'cosine') 
# MAGIC  * numFeatures for HashingTF (try 1000)

# COMMAND ----------

# add this to the pipeline to remove empty or short messages

emptyRowsRemover = SQLTransformer(statement='SELECT * FROM __THIS__ where size(noStopWords) >= 10')

# COMMAND ----------

tokenizer = Tokenizer(inputCol='message', outputCol='words')

stopWordsRemover = StopWordsRemover(inputCol='words', outputCol='noStopWords')

hashingTF = HashingTF(numFeatures=1000, inputCol='noStopWords', outputCol='hashingTF')

idf = IDF(inputCol='hashingTF', outputCol='idf')

normalizer = Normalizer(inputCol='idf', outputCol='features')

kmeans = KMeans(featuresCol='features', predictionCol='prediction', k=5, seed=1)

pipeline = Pipeline(stages=[tokenizer, stopWordsRemover, emptyRowsRemover, hashingTF, idf, normalizer, kmeans])

model = pipeline.fit(posts)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Apply the model on the data
# MAGIC 
# MAGIC hint
# MAGIC * just call transform, since the model is a transformer
# MAGIC * pass the training data as argument to the transform function

# COMMAND ----------

predictions = model.transform(posts)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### See how many pages are in your clusters
# MAGIC 
# MAGIC hint
# MAGIC * you can simply group by the column prediction and count
# MAGIC * the column with the cluster is called prediction by default

# COMMAND ----------

display(
  predictions
  .groupBy('prediction')
  .agg(count('*').alias('cnt'))
  .orderBy(desc('cnt'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### See what pages are in your clusters
# MAGIC 
# MAGIC hint
# MAGIC * just filter the result for specific cluster:
# MAGIC  * filter(col('prediction') == 0) and so on for other clusters

# COMMAND ----------

display(
  predictions
  .filter(col('prediction') == 0)
)

# COMMAND ----------

display(
  predictions
  .filter(col('prediction') == 1)
)

# COMMAND ----------

display(
  predictions
  .filter(col('prediction') == 2)
)

# COMMAND ----------

display(
  predictions
  .filter(col('prediction') == 3)
)

# COMMAND ----------

display(
  predictions
  .filter(col('prediction') == 4)
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC After playing a little bit with the data and input parameters to the learning algorithms, you might be able to identify that the data gets clustered according to language (with some error of course). By looking at some posts try to identify which clusters belong to english language and save it to a table. You can use this result as input in the next notebook where you will do LDA.

# COMMAND ----------

# we generate random string for the table name to avoid collisions
table_name = ''.join([random.choice('abcdefghijklmnoprstuvwxy') for _ in range(20)])

(
  predictions
  .select('page_id', 'message')
  .filter(col('prediction').isin([4])) # here write the number of clusters that belong to english language
  .repartition(32)
  .write
  .mode('overwrite')
  .saveAsTable(table_name)
)

print(table_name)

# COMMAND ----------


