import json
import boto3
import time
import numpy as np
import PIL.Image as Image
import tensorflow
from tensorflow.keras.models import load_model
from keras_metrics import *

NCHANNELS=3 #number of channels
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SHAPE = (IMAGE_HEIGHT,IMAGE_WIDTH,NCHANNELS)
f1= "model_shape_vgg16_model.h5"

customs_func = {"f1score": f1score}
model = load_model(f1, custom_objects=customs_func)

s3 = boto3.resource('s3')

def lambda_handler(event, context):
  bucket_name = event['Records'][0]['s3']['bucket']['name']
  key = event['Records'][0]['s3']['object']['key']

  img = readImageFromBucket(key, bucket_name).resize((IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
  img = np.array(img).reshape(1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])
  img = img/255
  X = [img, img]
  start_time = time.time()
  Y_ = model.predict(X)
  end_time = time.time() 
  results = np.argmax(Y_[0])
  print("ImageName %s: %s. Computation time: %s seconds" % (key,"positive" if results==1 else "negative", end_time - start_time))

def readImageFromBucket(key, bucket_name):
  bucket = s3.Bucket(bucket_name)
  object = bucket.Object(key)
  response = object.get()
  return Image.open(response['Body'])