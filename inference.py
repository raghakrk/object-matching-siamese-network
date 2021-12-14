import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_metrics import *
import PIL.Image as Image
import numpy as np
import json
import time

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
NCHANNELS=3 #number of channels
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SHAPE = (IMAGE_HEIGHT,IMAGE_WIDTH,NCHANNELS)

customs_func = {"f1score": f1score}
f1= "model_shape_vgg16_model.h5"
model = load_model(f1, custom_objects=customs_func)
img_path = '/mnt/c/Users/ragha/Documents/automotus_workspace/workspace/vehicle-rear/dataset/sm_prod/test_40/sm_xavier_agx_2230_main_st__2021-11-17T22_44_22.950Z__car__854dbd80__AttributeCaptureCrop__77e0de13__CONFIDENCE_SCORE__0.84__0.jpg'
img = Image.open(img_path).resize((IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
img = np.array(img).reshape(1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])
img = np.array(img)/255
X = [img, img]
start_time = time.time()
Y_ = model.predict(X)
end_time = time.time() 
results = np.argmax(Y_[0])
print("ImageName %s: %s. Computation time: %s seconds" % (img_path,"positive" if results==1 else "negative", end_time - start_time))

start_time = time.time()
Y_ = model.predict(X)
end_time = time.time() 
results = np.argmax(Y_[0])
print("ImageName %s: %s. Computation time: %s seconds" % (img_path,"positive" if results==1 else "negative", end_time - start_time))