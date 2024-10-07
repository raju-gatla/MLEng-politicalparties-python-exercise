# Databricks notebook source
import pandas as pd
import importlib
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('../src/text_loader')
from loader import DataLoader
from databricks.feature_store import FeatureStoreClient

data_loader = DataLoader(filepath='../data/Tweets.csv')
data_loader.data['Tweet'] = data_loader.data['Tweet'].astype(str)
vectorized_tweets = data_loader.preprocess_tweets()
encoded_parties = data_loader.preprocess_parties()
data_loader.data.drop_duplicates(subset=['Tweet'], inplace=True)


fs = FeatureStoreClient()
cleaned_df = spark.createDataFrame(data_loader.data)
fs.create_table(
    name='batch-1-mle.raju_gatla.cleaned_tweets',
    primary_keys='Tweet',
    df=cleaned_df,
    schema=cleaned_df.schema
)
