import os
import numpy as np
import matplotlib.pyplot as plt 
from keras.applications.vgg16 import VGG16
from utils import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

tvt='train'
videos = os.listdir('./images/' + tvt +'/')
for video in videos: 
    #video = videos[0]
    opath = './features/'+tvt +'/'+video+'/'
    if not os.path.exists(opath):
        os.makedirs(opath)
    frames = os.listdir('./images/' + tvt + '/' + video)
    for i in frames:
        #i= frames[0]
        frame = plt.imread('./images/' + tvt +'/'+ video + '/' + i)
        clip = preprocess_input(np.float32(frame))
        features = model.predict(np.expand_dims(clip, axis=0))
        features = np.squeeze(features)
        np.save(opath+i[:-4], features)
