from flask import Flask, render_template, request, Markup
from flask_cors import cross_origin
import numpy as np
import pandas as pd
import pickle
from utils.fertilizer import fertilizer_dic



crop_recommendation_model_path = 'model/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))


app = Flask(__name__)

#-----------------------------PAGES FUNCTION------------
@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")
    
@app.route("/crop_form")
@cross_origin()
def crop_form_func():
    return render_template("crop-Form.html")
    
@app.route('/fertilizer_form')
@cross_origin()
def fertilizer_form_func():

    return render_template('fertilizer.html')
    
    
#----------------PREDICTION FUNCTIONS------------------

@app.route("/crop-predict",endpoint='predict',methods=["GET","POST"])
@cross_origin()
def predict():
    if request.method=='POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('crop-result.html', prediction=final_prediction)

    return render_template('try-again.html')



@app.route('/fertilizer-predict', methods=["GET","POST"])
@cross_origin()
def fert_recommend():

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response)
    
    
    
if __name__ == '__main__':
    app.run(debug=False)