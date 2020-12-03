from flask import Flask,redirect,render_template,request

import pickle
import numpy as np
app=Flask(__name__)
import en_core_web_lg
nlp = en_core_web_lg.load()
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)
@app.route('/')
def hello_world():
    return render_template('index.html',text="Type message to check if any calamity")
@app.route('/predict',methods=["POST","GET"])
def predict():
    text_check=request.form['check']
    with nlp.disable_pipes():
        doc_vector = np.array([nlp(text_check).vector])
    output =model.predict(doc_vector)
    if output==1:
        return render_template('index.html',pred='It is emergency.Call 112',text=text_check)
    if output==0:
        return render_template('index.html',pred='Ahh False alarm.Relax',text=text_check)

app.run(debug=True, use_reloader=True)