import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# 1-184 falling down, 184-400 other
# every 8 video are from the same angle, e.g. 1,9,17
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

# np.random.permutation()
fall = [161, 128, 149, 133, 138, 124, 164,   0,  76,  77, 126, 167,  47,
        96,  56, 151,  59,  12,  68, 129, 142,  88,  25, 185,  48, 166,
       165,  40,  81,  66, 182, 147, 106,  16,  14,  11, 158,  15, 108,
        22, 123,  28,  30, 153,  79, 134,  89,  41,  36, 173, 104, 152,
        23, 155,  63,  10,  71, 116,  43,  50,   3, 135, 160,  49, 121,
         8, 169, 110, 109,  84,  17,  72,  34, 178,   2,  97, 137,  85,
        26, 162,  45, 117, 144, 136, 176,   6,  60,   5, 175, 141,  31,
        62,  37,  13,  74, 118,   4,  61,  32,  73, 159,  82, 184,  87,
        53,  99,  44, 115, 125, 186,  75, 103, 101, 156, 100,   7,  58,
        70, 143, 146, 114,  20, 119,  98,  21, 180,  54,  33,  93, 170,
        69,  46,   9, 105,  42, 139, 157, 168, 174,  65, 127, 179, 154,
       111, 132, 131, 172,  86,  35,  51, 122, 120, 140, 183,   1,  64,
       177, 163,  67,  24, 130, 113,  57, 150,  94,  39, 148,  83, 181,
        52,  38,  91,  80,  95,  29, 112,  90,  18, 145, 171,  92, 107,
        27,  55, 102,  19,  78]

other = [138, 122, 106, 192,  91, 116,  74,  51,  60, 110,  86, 145, 202,
       184, 155,  24, 120, 169, 173, 211,  56,  11,  69, 205, 137, 154,
       126,  64, 121, 183, 185, 125, 132, 210, 102,  70,  75,  93,  20,
        87, 144,  38,  98, 207,  85, 127, 142, 139,  16,  83, 147,   2,
        71,  43, 209,  52, 149, 196, 100,  53, 104, 180, 130,  36,  73,
       177,  79, 171,   0, 113,  63, 162, 131,  89, 134, 109,  97,   6,
       105, 108, 101, 156,  68, 117,  40,  94,  37, 124, 199, 140, 166,
        84,  27,  39,   8,  12,  99, 201,  72,  13,   5, 167, 114, 168,
        58, 212,  44, 165, 115, 112,  48,  25,  54,  95,  45, 136, 143,
        92,  47, 203, 103, 175,   4, 119, 208,  10, 128, 188, 133,  76,
       111, 206,  17, 181, 153,  50, 141, 172,  18,  35, 170,  55,  80,
       164, 193, 200, 151, 178,  14, 174,  61,   7, 157,  78,  96,  15,
        88, 190,   9,  28, 118,  42,  23,  90, 204, 161, 191,  30,  33,
       179, 163,  65,  34,  81, 187, 135,   3,  46, 182,  77,  31, 129,
       123,  19, 160,  26,  67, 152,  66, 176,  59, 194,  62, 189, 150,
       197, 148, 158, 198, 186, 159,  82,  22,   1,  41,  32,  21, 107,
       195, 146,  57,  49,  29]

num1=[93,28,65]  #187
num2=[107,32,75] #214

def loaddata(tvt):
	path = './dataset_falldetection/Multicamera/'

	if tvt is 'train':
		image = fall[:num1[0]]+other[:num2[0]]
		label = [0]*num1[0]+[1]*num2[0]
	elif tvt is 'val':
		image = fall[num1[0]:(num1[0]+num1[1])]+other[num2[0]:(num2[0]+num2[1])]
		label = [0]*num1[1]+[1]*num2[1]
	else:
		image = fall[(num1[0]+num1[1]):(num1[0]+num1[1]+num1[2])]+ \
			other[(num2[0]+num2[1]):(num2[0]+num2[1]+num2[2])]
		label = [0]*num1[2]+[1]*num2[2]
	X=[]
	for i in image:
		temp = []
		imgs=os.listdir(path+str(i+1).zfill(3))
		for name in imgs:
			temp.append(plt.imread(path+str(i+1).zfill(3)+'/'+name))
		temp = np.float32(temp)
		X.append(preprocess_input(temp))

	return np.array(X), to_categorical(label)

#x,y=loaddata('train') 










