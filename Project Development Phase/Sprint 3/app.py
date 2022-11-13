import requests
import os
from keras.preprocessing import image
from keras.models import load_model
from keras.utils import load_img,array_to_img
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template,redirect,url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app=Flask(__name__)
model=load_model(r'C:\Users\thani_k\Downloads\Fertilizers_Recommendation_System_For_Disease_Prediction\Sprint-2\Model Building\vegdata.h5')
model1=load_model(r'C:\Users\thani_k\Downloads\Fertilizers_Recommendation_System_For_Disease_Prediction\Sprint-2\Model Building\fruitdata.h5')
#home page
@app.route('/', methods=['GET', 'POST'])
def home():
	return render_template('home.html')

@app.route('/prediction')
def prediction():
	return render_template('predict.html')
@app.route('/predict',methods=['POST','GET'])

def predict():
	if(request.method=='POST'):
		f=request.files['image']
		basepath=os.path.dirname(__file__)
		file_path=os.path.join(basepath,'',secure_filename(f.filename))
		#print(file_path)
		f.save(file_path)
		img=tf.keras.utils.load_img(file_path,target_size=(128,128))
		x=tf.keras.utils.img_to_array(img)
		x=np.expand_dims(x,axis=0)
		plant=request.form['plant']
		print(plant)
		if(plant=='Vegetable'):
			preds=model.predict(x)
			print(preds)
			df=pd.read_excel(r'C:\Users\thani_k\Downloads\Fertilizers_Recommendation_System_For_Disease_Prediction\Sprint-3\precautions_veg.xlsx')
			#print(df.iloc[preds[0]]['caution'])
		else:
			preds=model1.predict(x)
			print("name=",preds[0])
			df=pd.read_excel(r'C:\Users\thani_k\Downloads\Fertilizers_Recommendation_System_For_Disease_Prediction\Sprint-3\precautions_fruits.xlsx')
			#print(df.iloc[preds[0]]['caution'])
			
		return render_template("predicted.html",disease=str(df.iloc[np.where(preds[0]==1)[0][0]]['disease']),data=str(df.iloc[np.where(preds[0]==1)[0][0]]['caution']))
        
if(__name__=="__main__"):
	app.run(debug=True)