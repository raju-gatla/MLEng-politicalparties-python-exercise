# Databricks notebook source
from pyspark.sql.functions import struct, col
import mlflow.pyfunc

mlflow.set_registry_uri('databricks-uc')
model_name = "batch-1-mle.raju_gatla.cleaned_tweets"
model_prod_uri = f"models:/{model_name}@Production"
prod_model = mlflow.sklearn.load_model(model_prod_uri)


# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Read cleaned dataset from Unity Catalog Feature Store
cleaned_df = fs.read_table('batch-1-mle.raju_gatla.cleaned_tweets').toPandas()

# Ensure the 'Tweet' column is of type string
cleaned_df['Tweet'] = cleaned_df['Tweet'].astype(str)

# Assuming 'model' is already defined and trained
predictions = prod_model.predict(cleaned_df['Tweet'])

# Display the predictions
display(predictions)
