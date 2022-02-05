from flask import Flask, render_template, request
#from keras.applications import ResNet50

from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)
model_path = 'D:\AI\Computer-Vision_2021\project_casting-product\model_predict.h5'
##resnet = ResNet50(weights='imagenet',input_shape=(224,224,3),pooling='avg')
#print("+"*50, "Model is loaded")
model = load_model(model_path)

#labels = pd.read_csv("labels.txt", sep="\n").values


def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():

	img = request.files['img']

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (300,300))

	image = np.reshape(image, (1,300,300,3))

	pred = model.predict(image)

	pred = np.argmax(pred)
	result = ['Ok_data', 'Error_data']
	if pred < 0.5:
		print(result[0])
	else:
		print(result[1])
	

	return render_template("prediction.html", data=pred)


if __name__ == "__main__":
	app.run(debug=True)
