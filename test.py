import joblib
from joblib import dump, load 
from sklearn import externals
import sklearn
from flask import Flask, request, render_template

clf = joblib.load('result.joblib.z')
classification = {0: "haineux", 1: "offensant", 2: "y'a r"}

app = Flask(__name__)



@app.route("/", methods=["GET", "POST"])
def index():
    tweet_type = -1
    text = ""
    if request.method == 'POST':
        text = request.form.get('t')
        tweet_type = classification[get_type(text)]
    return render_template("simple.html",t = text , type=tweet_type)
 
def get_type(sentence):
    prediction = clf.predict([sentence])
    print(prediction)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")