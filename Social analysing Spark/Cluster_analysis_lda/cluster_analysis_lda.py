# Databricks notebook source
# MAGIC %md
# MAGIC # Cluster Analysis and topic modelling using LDA
# MAGIC 
# MAGIC ## Task
# MAGIC Cluster the posts using LDA (Latent Dirichlet Allocation)
# MAGIC 
# MAGIC ## Data
# MAGIC * Take the same data that was used with KMeans - posts on facebook pages, but take only the cluster that corresponds to english pages
# MAGIC 
# MAGIC ## Notes
# MAGIC * Use LDA instead of KMeans
# MAGIC * You may want to play with number of topics and the size of vocabulary (the default size of CountVectorizer is 262144)
# MAGIC * You may want to do some more preprocessing of the text
# MAGIC  * for instance remove punctuation
# MAGIC  * or add some more words on the list provided to the StopWordsRemover
# MAGIC 
# MAGIC 
# MAGIC ## About LDA
# MAGIC * for more details about LDA see <a target="_blank" href="https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation">wiki</a>
# MAGIC * LDA model assumes that each document (post message in our case) is composed of some topics (number of these topics has to specified as input parameter)
# MAGIC * Each of these topics can be characterized by a set of words (bellow we provide a udf get_words that allows you to see the words to each topic)
# MAGIC * For each document you will get a topic distribution (a probability or weight for each topic in the document)
# MAGIC * The most probable topic in the document can be interpreted as cluster (bellow we provide a udf get_cluster that gives you index of the most probable topic)

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
# MAGIC ### Import functions and modules

# COMMAND ----------

from pyspark.sql.functions import col, count, desc, array_contains, split, explode, regexp_replace, lit

from pyspark.sql.types import ArrayType, StringType

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer, CountVectorizer

from pyspark.ml.clustering import LDA

from pyspark.ml import Pipeline


import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data
# MAGIC 
# MAGIC hint
# MAGIC * here we will use the dataset that you saved in the previous notebook so copy the table_name and use it here

# COMMAND ----------

# take the generated name from the previous notebook:
table_name = 'muutodfmuwfcmpxjfvwy'

data = spark.table(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore the data
# MAGIC 
# MAGIC hint
# MAGIC * see how many records you have

# COMMAND ----------

data.count()

# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Remove punctuation
# MAGIC 
# MAGIC hint
# MAGIC * it seems to be reasonable to do some more preprocessing on the data - one of the steps is removing the punctuation
# MAGIC * you can use <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.regexp_replace">regexp_replace</a> function of DF API
# MAGIC * you may try to use this (or some similar) regular expression: "[(.|?|,|:|;|!|>|<)]"

# COMMAND ----------

reg = "[(.|?|,|:|;|!|>|<)]"

pages = data.withColumn('message', regexp_replace('message', reg, ' '))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### See how many words you have in total in your documents
# MAGIC 
# MAGIC hint
# MAGIC * use functions <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.split">split</a> and <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.explode">explode</a> on the message field
# MAGIC * select the exploded message field and call <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.distinct">distinct</a> on it (or use <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dropDuplicates">dropDuplicates</a> equivalently)
# MAGIC * count number of rows

# COMMAND ----------

(
  pages
  .withColumn('words', split('message', ' '))
  .select(explode('words').alias('word'))
  .distinct()
  .count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Construct the pipeline
# MAGIC 
# MAGIC hint
# MAGIC * do vector representation for the texts
# MAGIC  * use: 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Tokenizer">Tokenizer</a> 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StopWordsRemover">StopWordsRemover</a> 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.CountVectorizer">CountVectorizer</a>
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.IDF">IDF</a> 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Normalizer">Normalizer</a> 
# MAGIC  * <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.LDA">LDA</a>
# MAGIC * you will have to choose number of topics for the LDA
# MAGIC * See the slides 83, 84, 85, 101 in the presentation
# MAGIC 
# MAGIC Notes
# MAGIC * with KMeans we used HashingTF to compute the term frequency as input for IDF
# MAGIC * here we are using countVectorizer so we can work with actual words and see how the topics are described later on

# COMMAND ----------

tokenizer = Tokenizer(inputCol='message', outputCol='words')

stopWordsRemover = StopWordsRemover(inputCol='words', outputCol='noStopWords')

countVectorizer = CountVectorizer(vocabSize=1000, inputCol='noStopWords', outputCol='tf', minDF=1)

idf = IDF(inputCol='tf', outputCol='idf')

normalizer = Normalizer(inputCol='idf', outputCol='features')

lda = LDA(k=7, maxIter=10)

pipeline = Pipeline(stages=[tokenizer, stopWordsRemover, countVectorizer, idf, normalizer, lda])

model = pipeline.fit(pages)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Apply the model on the data
# MAGIC 
# MAGIC hint
# MAGIC * just call transform, since the model is a transformer
# MAGIC * pass the training data as argument to the transform function

# COMMAND ----------

predictions = model.transform(pages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## See the result of LDA
# MAGIC 
# MAGIC hint
# MAGIC * select name, message, topicDistribution to see the probabilities for each topic in given document

# COMMAND ----------

display(
  predictions
  .select('message', 'topicDistribution')
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Helper functions (udfs)

# COMMAND ----------

# Some useful UDFs that will help you to do the next tasks

# vocabulary your model is using:
vocab = model.stages[2].vocabulary

# udf to extract the words for the topics
@udf(ArrayType(StringType()))
def get_words(termIndices):
  return [vocab[idx] for idx in termIndices]


# udf to determine the main topic for the document
@udf('integer')
def get_cluster(vec):
  return int(np.argmax(vec))


# udf to get the probability of a given topic in the document
@udf('double')
def get_topic_probability(vec, topic):
  return float(vec[topic])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Describe topics
# MAGIC 
# MAGIC hint
# MAGIC * each topic is characterized by a set of words
# MAGIC * use <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.LDAModel.describeTopics">describeTopics()</a> method of the LDA model to get the indices of the words in your vocabulary (model.stages[n].describeTopics(), here n is the index of LDA in your pipeline)
# MAGIC * use the udf get_words to see the actual words

# COMMAND ----------

display(
  model.stages[5].describeTopics()
  .withColumn('x', get_words(col('termIndices')))
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Find the most likely topic for each document
# MAGIC 
# MAGIC hint
# MAGIC * add new column named 'cluster' using the udf get_cluster to get the most likely topic for each post
# MAGIC * as argument for the udf use column topicDistribution which the result of LDA. This column contains vector with probabilities for each topic in the post
# MAGIC * you can now groupBy this new column and count how many posts are in given cluster

# COMMAND ----------

display(
   predictions
  .select('page_id', 'topicDistribution', 'message')
  .withColumn('cluster', get_cluster('topicDistribution'))
  .groupBy('cluster')
  .count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Order the documents by probability of specific topic
# MAGIC 
# MAGIC hint
# MAGIC * choose a topic index (for example 0)
# MAGIC * add new column called 'topicProbability' and extract here the probability your selected topic
# MAGIC  * these probabilities are in the column topicDistribution
# MAGIC  * to extract the probability you can use udf get_topic_probability implemented above. Just pass in the column topicDistribution and the index of your selected topic (you have to use the lit function for the topic index, for example: lit(0))
# MAGIC * order the DataFrame in descending order by this new column topicProbability

# COMMAND ----------

display(
   predictions
  .select('page_id', 'topicDistribution', 'message')
  .withColumn('topicProbability', get_topic_probability(col('topicDistribution'), lit(0)))
  .orderBy(desc('topicProbability'))
)
