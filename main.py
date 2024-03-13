from flask import Flask ,render_template ,request
import pandas as pd
import pickle
import numpy as np
app=Flask(__name__,template_folder='template')
data=pd.read_csv('cleaned_data.csv')
pipe=pickle.load(open("RegressionModel.pkl","rb"))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template("index1.html",locations=locations)
@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    BHK=request.form.get('BHK')
    bath=request.form.get('bath')
    sqft=request.form.get('total_sqft')
    print(location,BHK,bath,sqft)
    input = pd.DataFrame([[location,sqft,bath,BHK]],columns=['location','total_sqft','bath','BHK'])
    prediction=pipe.predict(input)[0] * 1e5


    return str(np.round(prediction,2))
if __name__=="__main__" :
    app.run(debug=True,port=5000)
