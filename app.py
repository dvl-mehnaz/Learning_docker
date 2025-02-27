from flask import Flask, request
import pandas as pd
import pickle

app=Flask(__name__)
pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "welcome all"


@app.route('/predict')
def model_prediction():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The Prediction value is " + str(prediction)


@app.route("/test_file",methods=['POST'])
def predict_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The prediction file for cvs is :" + str(list(prediction))



if __name__=="__main__":
    app.run(debug=True)