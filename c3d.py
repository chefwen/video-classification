import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time

from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.callbacks import ModelCheckpoint, CSVLogger

model_name = '3dcnn'

num_cuts = 5
nb_epoch = 50
saved_model = None#'./result/svgg.038-1.329.hdf5'
nb_classes = 5
img_size = [240, 320, 3]

class mymodels():
	def __init__(self, nb_classes, modelname, num_cuts, img_size=[240,320,3], saved_model=None):
		self.load_model = load_model
		self.saved_model = saved_model
		self.nb_classes = nb_classes
		self.num_cuts = num_cuts

		metrics = ['accuracy']
		if self.saved_model is not None:
			print("Loading model %s" % self.saved_model)
			self.model = load_model(self.saved_model)
			print(self.model.summary())
		elif modelname == '3dcnn':
			print("Loading Conv3D")
			self.input_shape = [num_cuts]+img_size
			self.model = self.c3d1()
		else:
			print("Unknown network!")
			sys.exit()

		#optimizer = Adam(lr=1e-4)
		optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
			metrics=metrics)

	def c3d1(self):
		model = Sequential()
		model.add(Conv3D(16,(3,3,3), padding='same', input_shape=
			self.input_shape, activation='relu'))
		model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
		model.add(Conv3D(16, (3,3,3), padding='same', activation='relu'))
		model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
		model.add(Conv3D(16, (3,3,3), padding='same', activation='relu'))
		#model.add(Conv3D(16, (3,3,3), padding='same', activation='relu'))
		model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
		model.add(Conv3D(16, (3,3,3), padding='same', activation='relu'))
		#model.add(Conv3D(16, (3,3,3), padding='same', activation='relu'))
		model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
		model.add(Conv3D(16, (3,3,3), padding='same', activation='relu'))
		#model.add(Conv3D(512, (3,3,3), padding='same', activation='relu'))
		model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
		model.add(Flatten())
		model.add(Dropout(0.5))
		model.add(Dense(215))
		model.add(Dropout(0.5))
		model.add(Dense(self.nb_classes, activation='softmax'))
		print(model.summary())

		return model

def preprocess_input(x, dim_ordering='default'):
    # Zero-center by mean pixel
    #r,g,b = rgb['tvt']test(102.99,95.43,83.72)
    if x.ndim==4:
        x[:, :, :, 0] -= 123.68
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 103.939
    else:
        x[:, :, 0] -= 123.68
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 103.939
    return x/128

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    s = [10, 18, 48, 86, 94]
    #s = np.unique(y)
    for i in range(len(s)):
        y[y==s[i]]=i
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes),dtype=np.int8)
    categorical[np.arange(n), y] = 1
    return categorical

def load_all(tvt, num_cuts, test=False):
    class_index = {'TrampolineJumping': 94, 'BoxingSpeedBag': 18, 'BenchPress': 10, 'JumpRope': 48, 'StillRings': 86}
    videos = os.listdir('./images/' + tvt +'/')
    X, y = [], []
    for video in videos: 
        frames = os.listdir('./images/' + tvt + '/' + video)
        frames = sorted(frames)
        inter = len(frames)/num_cuts
        clip = []
        for i in range(num_cuts):
            if test:
                num = int(i*inter +1)
            else:
                num = np.random.randint(i*inter, (i+1)*inter)
            frame = plt.imread('./images/' + tvt +'/'+ video + '/' 
                + frames[num])
            clip.append(frame)
        clip = np.float32(clip)
        clip = np.squeeze(clip)# reduce dim while num_cut=1
        X.append(preprocess_input(clip))
        y.append(class_index[video[2:-8]])
    tmp_inp = np.array(X)
    tmp_targets = to_categorical(y,5)
    return tmp_inp, tmp_targets


tdata = load_all('train', num_cuts=num_cuts)
vdata = load_all('val', num_cuts=num_cuts)
#sdata = load_all('test', num_cuts=num_cuts, True)

checkpointer = ModelCheckpoint(filepath='./result/'+model_name+\
    '.{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=True)

timestamp = time.time()
csv_logger = CSVLogger('./result/' + model_name + str(timestamp) + '.log')

mdl = mymodels(nb_classes, model_name, num_cuts, img_size, saved_model)
mdl.model.fit(x=tdata[0], y=tdata[1], 
    epochs=nb_epoch, callbacks=[checkpointer, csv_logger], 
    validation_data=vdata)

mdl.model.save('./result/final.hdf5')

# result = mdl.model.evaluate(sdata[0],sdata[1])
# print(result[1])






