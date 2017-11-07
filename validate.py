import numpy as np
import os
import matplotlib.pyplot as plt
from utils import preprocess_input
from keras.models import load_model

from utils import DataSet

# from keras import backend as K
# K.set_learning_phase(1) #set learning phase


num_cuts = 5
feature = True
model = load_model('./result/a/mcnn.030-0.127.hdf5')
batch_size = 1
data = DataSet('test', feature)
generator = data.generate_a(batch_size=batch_size, num_cuts=num_cuts)
steps = np.ceil(data.get_num()//batch_size)

result = model.evaluate_generator(generator, steps)
print(result[1])

def predict(tvt, video, num_cuts):
    class_index = {0:'BenchPress', 1:'BoxingSpeedBag', 2:'JumpRope',
        3:'StillRings', 4: 'TrampolineJumpling'}
    frames = os.listdir('./images/' + tvt + '/' + video)
    frames = sorted(frames)
    inter = len(frames)/num_cuts
    X = []
    clip = []
    for i in range(num_cuts):
        num = np.random.randint(i*inter, (i+1)*inter)
        frame = plt.imread('./images/' + tvt +'/'+ video + '/'
            + frames[num])
        #print video+frames[num]
        clip.append(frame)
    clip = np.float32(clip)
    clip = np.squeeze(clip)# reduce dim while num_cut=1
    X.append(preprocess_input(clip))
    res = model.predict(np.array(X))
    print(class_index[res.argmax()])


print model.summary()
#validate('train', 'v_BoxingSpeedBag_g09_c02', num_cuts=num_cuts)
