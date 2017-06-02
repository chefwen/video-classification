import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix

def preprocess_input(x):
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

class dataset():

    def __init__(self, tvt, feature = False):
        if feature:
            self.f = np.load
            self.path = './features/'+ tvt+'/'
        else:
            self.f = plt.imread
            self.path = './images/'+ tvt+'/' 
        self.class_index = {'BenchPress': 10, 'BoxingSpeedBag': 18, 'JumpRope': 48, 'StillRings': 86, 'TrampolineJumping': 94}
        self.videos = os.listdir(self.path)


    def generate(self, batch_size, num_cuts):
        X = []
        while True:
            n = 0
            for video in self.videos:
                n = n+1
                frames = os.listdir(self.path + video)
                frames = sorted(frames)
                inter = len(frames)/num_cuts
                clip = []
                for i in range(num_cuts):
                    num = int(i*inter +1)
                    frame = self.f(self.path + video + '/' + frames[num])
                    clip.append(frame)
                clip = np.float32(clip)
                clip = np.squeeze(clip)# reduce dim while num_cut=1
                X.append(preprocess_input(clip))      
                if len(X)==batch_size or n == len(self.videos):
                    tmp_inp = np.array(X)
                    X = []
                    yield tmp_inp

    def get_label(self):
        # tvt -dataset type, 'train', 'val', 'test'
        y = []
        for video in self.videos:
            y.append(self.class_index[video[2:-8]])
        s = [10, 18, 48, 86, 94]#specific for tiny UCF101
        y = np.array(y)
        for i in range(len(s)):
            y[y==s[i]]=i       
        return y

    def get_num(self):
        return len(self.videos)

    def get_name(self):
        return self.videos

batch_size = 10
num_cuts = 1 
model = load_model('./result/record/2d/dp/0.7pre.040-0.280.hdf5')
data = dataset('test', feature = False)
generator = data.generate(tvt='test', batch_size=batch_size, num_cuts=num_cuts)
steps = np.ceil(data.get_num()//batch_size)+1

result = model.predict_generator(generator, steps)
prediction = result.argmax(axis=1)
true_label = data.get_label()
matrix = confusion_matrix(true_label, prediction)
# index = [i for i in range(214) if prediction[i]==2 and true_label[i]==0]
# seven trampline as rope [69, 112, 113, 143, 162, 182, 193]
