import numpy as np
import tensorflow
from utils import resnet6, resnet8, resnet50_model, vgg16_model, GoogLeNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.utils.data_utils import Sequence
# from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras_metrics import *
from sklearn import metrics
import json
import albumentations as albu
import random
from sys import argv
from collections import Counter
import os

L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))
kmetrics={"class_output":['acc',f1score]}

NUM_EPOCHS = 10
nchannels=3 #number of channels
# image_size_w_c = 224 #image´s width for vehicle´s shape
# image_size_h_c = 224 #image´s height for vehicle´s shape
# batch_size = 8
image_size_h_c = 112
image_size_w_c = 112
batch_size = 32

def load_img(img, vec_size):
  iplt0 = process_load(img[0][0], vec_size)
  iplt1 = process_load(img[1][0], vec_size)

  d1 = {"i0":iplt0,
        "i1":iplt1,
        "l":img[2],
        "p1":img[0][0],
        "p2":img[1][0],
        }
  return d1

def process_load(f1, vec_size):
    _i1 = image.load_img(f1, target_size=vec_size)
    _i1 = image.img_to_array(_i1, dtype='uint8')
    return _i1

def fold(list1, ind, train=False):
    _list1 = list1.copy()
    _list1.pop(ind)
    if train:
        return [_list1[i % len(_list1)] for i in [ind+2,ind+3]]
    else:
        return [_list1[i % len(_list1)] for i in [ind+4,ind+5]]

class SiameseSequence(Sequence):
    def __init__(self,features, 
                augmentations,
                batch_size=batch_size,
                input2=(image_size_h_c,image_size_w_c,nchannels), 
                type1=None,
                metadata_dict=None, 
                metadata_length=0, 
                with_paths=False):
        self.features = features
        self.batch_size = batch_size
        self.vec_size2 = input2
        self.type = type1
        self.metadata_dict = metadata_dict
        self.metadata_length = metadata_length
        self.augment = augmentations
        self.with_paths = with_paths

    def __len__(self):
        return int(np.ceil(len(self.features) / 
            float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch = self.features[start:end]
        futures = []
        _vec_size = (len(batch),) + self.vec_size2
        b1 = np.zeros(_vec_size)
        b2 = np.zeros(_vec_size)
        blabels = np.zeros((len(batch)))
        p1 = []
        p2 = []
        if self.metadata_length>0:
            metadata = np.zeros((len(batch),self.metadata_length))

        i1 = 0
        for _b in batch:
            res = load_img(_b, self.vec_size2)
            if self.augment is not None:
                b1[i1,:,:,:] = self.augment[0][0](image=res['i0'])["image"]
                b2[i1,:,:,:] = self.augment[1][0](image=res['i1'])["image"]
            else:
                b1[i1,:,:,:] = res['i0']
                b2[i1,:,:,:] = res['i1']
            blabels[i1] = res['l']
            p1.append(res['p1'])
            p2.append(res['p2'])
            i1+=1
        blabels2 = np.array(blabels).reshape(-1,1)
        blabels = to_categorical(blabels2, 2)
        y = {"class_output":blabels, "reg_output":blabels2}
        result = [[b1, b2], y]
        if self.with_paths:
            result += [[p1,p2]]
            return result[0], result[1], result[2]
        else:
            return result[0], result[1]

def siamese_model(model, input2):
    left_input_C = Input(input2)
    right_input_C = Input(input2)
    convnet_car = model(input2)
    encoded_l_C = convnet_car(left_input_C)
    encoded_r_C = convnet_car(right_input_C)
    inputs = [left_input_C, right_input_C]

    # Add the distance function to the network
    x = L1_layer([encoded_l_C, encoded_r_C])
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, kernel_initializer='normal',activation='relu')(x)
    x = Dropout(0.5)(x)
    predF2 = Dense(2,kernel_initializer='normal',activation='softmax', name='class_output')(x)
    regF2 = Dense(1,kernel_initializer='normal',activation='sigmoid', name='reg_output')(x)
    optimizer = Adam(0.0001)
    losses = {
        'class_output': 'binary_crossentropy',
        'reg_output': 'mean_squared_error'
    }

    lossWeights = {"class_output": 1.0, "reg_output": 1.0}

    model = Model(inputs=inputs, outputs=[predF2, regF2])
    model.compile(loss=losses, loss_weights=lossWeights,optimizer=optimizer, metrics=kmetrics)

    return model

def calculate_metrics(ytrue1, ypred1):
    conf = metrics.confusion_matrix(ytrue1, ypred1, labels = [0,1])

    maxres = (conf[1,1],
              conf[0,0],
              conf[0,1],
              conf[1,0],
        metrics.precision_score(ytrue1, ypred1) * 100,
        metrics.recall_score(ytrue1, ypred1) * 100,
        metrics.f1_score(ytrue1, ypred1) * 100,
        metrics.accuracy_score(ytrue1, ypred1) * 100)
    return maxres

#------------------------------------------------------------------------------
def test_report(model_name, model, test_gen):
    print("=== Evaluating model: {:s} ===".format(model_name))
    a = open("inferences_output.txt", "w")
    ytrue, ypred = [], []
    for i in range(len(test_gen)):
      X, Y, paths = test_gen[i]
      Y_ = model.predict(X)
      for y1, yreg, y2, p0, p1 in zip(Y_[0].tolist(), Y_[1].tolist(), Y['class_output'].argmax(axis=-1).tolist(), paths[0], paths[1]):
        y1_class = np.argmax(y1)
        ypred.append(y1_class)
        ytrue.append(y2)
        a.write("%s;%s;%d;%d;%f;%s\n" % (p0, p1, y2, y1_class, yreg[0], str(y1)))

    a.write('tp: %d, tn: %d, fp: %d, fn: %d P:%0.2f R:%0.2f F:%0.2f A:%0.2f' % calculate_metrics(ytrue, ypred))
    a.close()
    print('tp: %d, tn: %d, fp: %d, fn: %d P:%0.2f R:%0.2f F:%0.2f A:%0.2f' % calculate_metrics(ytrue, ypred))

keys = ['bellevue', 'sm_1401', 'sm_2309', 'sm_2400', 'sm_2645', 'sm_319', 'sm_prod']
input2 = (image_size_h_c,image_size_w_c,nchannels)
# model = resnet6
model = GoogLeNet
name = 'GoogLeNet'
train_augs = [[],[]]
test_augs = [[],[]]
tam_max = 3
data = None
if os.name == 'nt':
    data = json.load(open('dataset_test_images_windows.json'))
else:
    data = json.load(open('dataset_test_images.json'))

seq_car = albu.Compose(
    [
        albu.IAACropAndPad(px=(0, 8)),
        albu.IAAAffine(
        scale=(0.8, 1.2),
                shear=(-8, 8),
        order=[0,1],
        cval=(0),
        mode='constant'),
        albu.ToFloat(max_value=255)
    ],p=0.7
)
AUGMENTATIONS_TEST = albu.Compose([
    albu.ToFloat(max_value=255)
])

for i in range(tam_max):
    train_augs[0].append(seq_car)
    train_augs[1].append(seq_car)
    test_augs[0].append(AUGMENTATIONS_TEST)
    test_augs[1].append(AUGMENTATIONS_TEST)

random.shuffle(keys)
val = data[keys[0]]
trn=[]
for i in range(1,len(keys)):
    trn += data[keys[i]]
trnGen = SiameseSequence(trn, train_augs,batch_size=batch_size,input2=input2, type1='car')
tstGen = SiameseSequence(val, test_augs,batch_size=batch_size,input2=input2, type1='car')
siamese_net = siamese_model(model, input2)

f1 = 'model_shape_%s.h5' % (name)

#fit model
history = siamese_net.fit_generator(trnGen,
                            epochs=NUM_EPOCHS,
                            validation_data=tstGen)
#validate plate model
tstGen2 = SiameseSequence(val, None, batch_size=batch_size,input2=input2, with_paths=True, type1='car')
test_report('validation_shape_%s' % name,siamese_net, tstGen2)
siamese_net.save(f1)