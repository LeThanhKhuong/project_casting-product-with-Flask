import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
model_path = 'D:\AI\Computer-Vision_2021\project_casting-product\model_predict.h5'
##resnet = ResNet50(weights='imagenet',input_shape=(224,224,3),pooling='avg')
#print("+"*50, "Model is loaded")

model = load_model(model_path)


image = cv2.resize(image, (300,300))

image = np.reshape(image, (1,300,300,3))

pred = model.predict(image)

pred = np.argmax(pred)
if pred < 0.5:
    print("def_front")
    cv2.putText(image, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    print("ok_front")
    cv2.putText(image, "ok_front",(10, 30),    cv2.FONT_HERSHEY_SIMPLEX,   0.6, (0, 255, 0), 2) 
plt.axis(pred, cmap = 'gray')
plt.show()


#https://github.com/JustinSmethers/Keras-CNN-Image-Classification---Cats-and-Dogs/blob/master/Keras%20CNN%20Image%20Classification%20-%20Cats%20and%20Dogs.py
#https://github.com/sagar448/Keras-Convolutional-Neural-Network-Python
