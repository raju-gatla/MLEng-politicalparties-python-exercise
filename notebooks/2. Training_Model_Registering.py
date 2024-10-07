# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

cleaned_df = fs.read_table('batch-1-mle.raju_gatla.cleaned_tweets').toPandas()
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_df['Tweet'], cleaned_df['Party'], test_size=0.2, random_state=42)


# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------


model_name = f"batch-1-mle.raju_gatla.cleaned_tweets"
mlflow.set_experiment("/Users/raju.gatla@thoughtworks.com/my_experiments")

# COMMAND ----------

from mlflow.models import infer_signature
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


X_train = X_train.astype(str)

with mlflow.start_run(run_name="Party-Tweets-Model_Demo"): 


    model = make_pipeline(TfidfVectorizer(), MultinomialNB())


    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True
    )
    
    model.fit(X_train, y_train)


    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=model_name
    )
    client.set_registered_model_alias(model_name, "Staging", get_latest_model_version(model_name))

# COMMAND ----------


model_details = client.get_registered_model(model_name)
display(model_details)

# COMMAND ----------


