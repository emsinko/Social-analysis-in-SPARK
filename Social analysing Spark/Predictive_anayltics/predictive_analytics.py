# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Create predictive model
# MAGIC <br>
# MAGIC ## Task
# MAGIC Construct model that is going to predict if an influencer is going to publish a post next day or not. Model it as binary classification.
# MAGIC 
# MAGIC ## Data
# MAGIC * use two datasets about influencers
# MAGIC * the first dataset contains basic information about each influencer
# MAGIC * the second dataset contains posting history for each influncer for the past 6 months
# MAGIC 
# MAGIC ## Notes
# MAGIC * the posting history is for the period 1.1.2018 - 1.8.2018
# MAGIC * assume it is 31.7.2017 and make a prediction for the next day
# MAGIC * extract the labels for 1.8. to constract the training and test dataset
# MAGIC * extract some features from the available data
# MAGIC * experiment with these models: Logistic regression (lr), decision tree (dt), random forest (rf)
# MAGIC * Try to construct some basic model first and than improve it by adding some more features

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

from pyspark.sql.functions import col, max, datediff, count, desc, array_contains, broadcast, explode, length, first, when, expr, regexp_replace, row_number, coalesce, lit, coalesce, size

from pyspark.sql import Window


from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

infl = spark.table('mlprague.influencers')

posts_history = spark.table('mlprague.infl_posting_history')

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

# your code here:
infl.count()

# COMMAND ----------

posts_history.count()

# COMMAND ----------

infl.printSchema()

# COMMAND ----------

posts_history.printSchema()

# COMMAND ----------

display(infl)

# COMMAND ----------

display(posts_history)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract the label
# MAGIC 
# MAGIC hint:
# MAGIC * use the posts history dataset and see what influencers posted on 1.8.2018 and assign them label 1
# MAGIC  * use withColumn() transormation together with lit(1) which adds a column with constant value 1
# MAGIC  * see lit() function in <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.lit">docs</a> with example
# MAGIC * left join this on the influencers and those records with null value will have label 0

# COMMAND ----------

label = (
  posts_history
  .filter(col('post_date') == '2018-08-01')
  .select('influencer_id')
  .distinct()
  .withColumn('label', lit(1))
)

influencers_with_label = (
  infl
  .join(label, 'influencer_id', 'left')
  .withColumn('label', coalesce('label', lit(0)))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### You may also want to check how many datapoints you have for each class
# MAGIC 
# MAGIC hint
# MAGIC * use groupBy('label').count()

# COMMAND ----------

display(
  influencers_with_label
  .groupBy('label')
  .agg(count('*').alias('ct'))
)

# COMMAND ----------

display(influencers_with_label)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Construct some basic features
# MAGIC 
# MAGIC hint:
# MAGIC * you may try number of interests, number of languages, age
# MAGIC * interests and language cols are of ArrayType
# MAGIC  * you can use <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.size">size</a> function to count number of its elements
# MAGIC  * the slide 48 in the prezentation might be useful for using functions on arrays

# COMMAND ----------

data_with_basic_features = (
  influencers_with_label
  .withColumn('num_interests', size('interests'))
  .withColumn('num_languages', size('languages'))
)

# COMMAND ----------

display(data_with_basic_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the data for training and testing
# MAGIC 
# MAGIC hint
# MAGIC * use the function <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit">randomSplit</a>
# MAGIC * see the slide 99 in the presentation

# COMMAND ----------

(train, test) = data_with_basic_features.randomSplit([0.7, 0.3], 24)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Construct & fit the pipeline
# MAGIC 
# MAGIC hint:
# MAGIC * use <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler">VectorAssembler</a> to create the input features 
# MAGIC * choose your model 
# MAGIC  * for LR use <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier">RandomForestClassifier</a> 
# MAGIC  * for RF use <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression">LogisticRegression</a> 
# MAGIC  * for DT use <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.DecisionTreeClassifier">DecisionTreeClassifier</a> 
# MAGIC * the slide 104 in the prezentation might be useful for constructing the pipeline
# MAGIC * use train data for training

# COMMAND ----------

# features:
features_array = ['num_interests', 'num_languages']

# Assambler:
assembler = VectorAssembler(inputCols=(features_array), outputCol='features')

# Classifier:
rf = RandomForestClassifier(labelCol='label', featuresCol='features', seed=42)

pipeline = Pipeline(stages=[assembler, rf])

rf_model = pipeline.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Evaluate the model
# MAGIC 
# MAGIC hint: 
# MAGIC * use the <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator">BinaryClassificationEvaluator</a> 
# MAGIC * the slide 106 in the prezentation might be useful for evaluating binary classification
# MAGIC * use the test data for evaluation

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

predictions = rf_model.transform(test)

evaluator.evaluate(predictions)

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The accuracy is not very great. Perhaps we can improve it by some more predictors

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Try to improve the model
# MAGIC 
# MAGIC hint:
# MAGIC * you may try also some categorical features like the value of the interest
# MAGIC * the slide 88, 94 in the prezentation might be useful for OneHotEncoder and StringIndexer

# COMMAND ----------

data_with_catagorical_feature = (
  data_with_basic_features.withColumn('interest', col('interests')[0])
)

# COMMAND ----------

display(data_with_catagorical_feature)

# COMMAND ----------

(train, test) = data_with_catagorical_feature.randomSplit([0.7, 0.3], 24)

# COMMAND ----------

# features:

features_array = ['num_interests', 'num_languages']

# indexer
interestIndexer = StringIndexer(inputCol='interest', outputCol='indexedInterest')

# OneHotEncoders:
interestEncoder = OneHotEncoder(inputCol='indexedInterest', outputCol='interestVec')

# Assambler:
assembler = VectorAssembler(inputCols=(features_array + ['interestVec']), outputCol='features')

# Classifier:
rf = RandomForestClassifier(featuresCol='features', seed=42)

pipeline = Pipeline(stages=[interestIndexer, interestEncoder, assembler, rf])

rf_model = pipeline.fit(train)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

predictions = rf_model.transform(test)

evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC The accuracy is slightly better but still not very good. Let's see if we can improve it even better:

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Improve the model even more
# MAGIC 
# MAGIC hint:
# MAGIC * construct some features that capture how frequently the influencer posts
# MAGIC * extract these features from the posting history

# COMMAND ----------

history_for_features = (
  posts_history
  .filter(col('post_date') <= '2018-07-31')
)

time_from_last_post = (
  history_for_features
  .groupBy('influencer_id')
  .agg(
    max('post_date').alias('last_post')
  )
  .withColumn('time_from_last_post', datediff(lit('2018-07-31'), col('last_post')))
  .select('influencer_id', 'time_from_last_post')
)

number_of_posts = (
  history_for_features
  .groupBy('influencer_id')
  .agg(
    count('*').alias('number_of_posts')
  )
)

# COMMAND ----------

data_features_improved = (
  data_with_catagorical_feature
  .join(time_from_last_post, 'influencer_id')
  .join(number_of_posts, 'influencer_id')
)

# COMMAND ----------

display(data_features_improved)

# COMMAND ----------

(train, test) = data_features_improved.randomSplit([0.7, 0.3], 24)

# COMMAND ----------

# features:
features_array = ['num_interests', 'num_languages', 'time_from_last_post', 'number_of_posts']

# indexer
interestIndexer = StringIndexer(inputCol='interest', outputCol='indexedInterest')

# OneHotEncoders:
interestEncoder = OneHotEncoder(inputCol='indexedInterest', outputCol='interestVec')

# Assambler:
assembler = VectorAssembler(inputCols=(features_array), outputCol='features')

# Classifier:
rf = RandomForestClassifier(featuresCol='features', seed=42)

pipeline = Pipeline(stages=[interestIndexer, interestEncoder, assembler, rf])

rf_model = pipeline.fit(train)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

predictions = rf_model.transform(test)

evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Try crossvalidation
# MAGIC 
# MAGIC hint
# MAGIC * the slide 108 in the prezentation might be useful for tunning hyperparameters
# MAGIC * check in the documentation what parameters has your model (maxDepth, numTrees for Random Forrest)

# COMMAND ----------

paramGrid = (
  ParamGridBuilder()
  .addGrid(rf.maxDepth, [3, 5, 8])
  .addGrid(rf.numTrees, [50, 100, 150])
  .build()
)

cross_model = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid).fit(train)

rf_model = cross_model.bestModel

# COMMAND ----------

predictions = rf_model.transform(test)
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## See some properties of the final model
# MAGIC 
# MAGIC Note
# MAGIC * This depends on the model you are using
# MAGIC 
# MAGIC Hint
# MAGIC * For Random Forest see the API of the model in <a target="_blank" href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassificationModel">docs</a>
# MAGIC * Example to see number of trees:
# MAGIC  * rf_model.stages[n].getNumTrees and here rf_model is your trained model and n is index of RF in your pipeline

# COMMAND ----------

rf_model.stages[3].getNumTrees

# COMMAND ----------

rf_model.stages[3].totalNumNodes

# COMMAND ----------

rf_model.stages[3].trees

# COMMAND ----------

rf_model.stages[3].toDebugString

# COMMAND ----------


