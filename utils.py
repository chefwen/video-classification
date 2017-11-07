"""
temperal, only works for two classes
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import random

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
    s = [10, 18, 48, 86, 94]#specific for tiny UCF101
    for i in range(len(s)):
        y[y==s[i]]=i
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes),dtype=np.int8)
    categorical[np.arange(n), y] = 1
    return categorical



class DataSet():
    def __init__(self, tvt, feature = False):
        # tvt -dataset type, 'train', 'val', 'test'
        if feature:
            self.f = np.load
            self.path = './features/'+ tvt+'/'
        else:
            self.f = plt.imread
            self.path = './images/'+ tvt+'/' 
        self.class_index = {'BenchPress': 10, 'BoxingSpeedBag': 18, 'JumpRope': 48, 'StillRings': 86, 'TrampolineJumping': 94}
        self.videos = os.listdir(self.path)

    def get_classes(self):
        f=open("./ucfTrainTestlist/classInd.txt","r")
        videos = f.readlines()
        f.close
        videos = [video.strip() for video in videos]
        class_index = {}
        for video in videos:
            video_s = video.split()
            class_index[str(video_s[1])] = int(video_s[0])
        return class_index

    def generate(self, batch_size, num_cuts):
        while True:
            X, y = [], []
            for _ in range(batch_size):
                video = random.choice(self.videos) 
                frames = os.listdir(self.path + video)
                frames = sorted(frames)
                inter = len(frames)/num_cuts
                clip = []
                for i in range(num_cuts):
                    num = np.random.randint(i*inter, (i+1)*inter)
                    frame = self.f(self.path+ video + '/' 
                        + frames[num])
                    clip.append(frame)
                clip = np.squeeze(clip)
                X.append(clip)
                y.append(self.class_index[video[2:-8]])

            tmp_inp = np.array(X)
            tmp_targets = to_categorical(y, 5)
            yield tmp_inp, tmp_targets


    def generateall(self):
        while True:
            X, y = [], []
            n=0
            for video in self.videos: 
                n=n+1
                if n==len(self.videos):
                    print('Reach the end')
                frames = os.listdir(self.path + video)
                frames = sorted(frames)
                clip = []
                for i in frames:
                    frame = self.f(self.path+ video + '/' 
                        + i)
                    clip.append(frame.flatten())
                #clip = np.amax(clip, axis=0)#comment if not all mcnn case
                X.append(clip)
                y.append(self.class_index[video[2:-8]])

                tmp_inp = np.array(X)
                tmp_targets = to_categorical(y, 5)
                X, y = [], []
                yield tmp_inp, tmp_targets


    def generate_a(self, batch_size, num_cuts):
        X, y = [], []
        while True:
            n = 0
            for video in self.videos:
                n = n+1
                frames = os.listdir(self.path+ video)
                frames = sorted(frames)
                inter = len(frames)/num_cuts
                clip = []
                for i in range(num_cuts):
                    num = int(i*inter +1)
                    frame = self.f(self.path+ video + '/' 
                        + frames[num])
                    clip.append(frame)
                clip = np.squeeze(clip) # reduce dim while num_cut=1
                X.append(clip)
                y.append(self.class_index[video[2:-8]])       
                if len(y)==batch_size or n == len(self.videos):
                    if n == len(self.videos):
                        print("Reach the end")
                    tmp_inp = np.array(X)
                    tmp_targets = to_categorical(y,5)
                    X, y = [], []
                    yield tmp_inp, tmp_targets

    def load_all(self, tvt, num_cuts):
        class_index = self.get_classes()
        videos = os.listdir('./images/' + tvt +'/')
        X, y = [], []
        for video in videos: 
            frames = os.listdir('./images/' + tvt + '/' + video)
            frames = sorted(frames)
            inter = len(frames)/num_cuts
            clip = []
            for i in range(num_cuts):
                num = int(i*inter +1)
                #num = np.random.randint(i*inter, (i+1)*inter)
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

    def get_num(self):
        return len(self.videos)

if __name__ =='__main__':
    # check if the image has the right label
    data=DataSet('val', False) 
    b= data.generate_a(50, 3)
    a,c =next(b)
    s = [10, 18, 48, 86, 94]
    num = 0
    plt.imshow(a[num,0,:,:,:].astype('uint8'))
    temp = data.get_classes()
    label = [aa for aa in temp if temp[aa]==np.sum(s*c[num,:])]
    print(label[0])
    plt.show()
