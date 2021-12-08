from flask import Flask, render_template, jsonify
import json ,os
import joblib 
import pandas as pd
from sklearn.pipeline import make_pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import externals
from stop_words import get_stop_words
import html , re
from joblib import dump, load


df = pd.read_csv('/home/ubuntu/Desktop/test/data/labels.csv', delimiter=",")
df = df.drop(columns='Unnamed: 0')
df = df.drop(columns='count')
df["tweet"] = df['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', '’ ', tweet.lower()))

clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)

clf = clf.fit(X=df["tweet"], y=df["class"])
dump(clf, 'result.joblib') 


