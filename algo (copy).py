import html

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


df = pd.read_csv('/home/ubuntu/Desktop/test/data/labels.csv', delimiter=",")
df = df.drop(columns='Unnamed: 0')
df = df.drop(columns='count')
tweets = df["tweet"].transform(lambda line: html.unescape(line))
X = tweets
y = df[ ["class"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.50, random_state=413)
clf = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)

clf = clf.fit(X_train,y_train)

model_filename = 'result.joblib.z'

joblib.dump(clf, model_filename)
