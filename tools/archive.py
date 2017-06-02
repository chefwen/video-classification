import os
import matplotlib.pyplot as plt
import numpy as np

def find_len():
	len_video = []
	for tvt in ['train', 'val', 'test']:
		videos = os.listdir('./images/'+tvt)
		for video in videos:
			temp = os.listdir('./images/'+tvt+'/'+video)
			len_video.append(len(temp))

	print("The smallest frame length is %d ." %min(len_video))#45

def showfeature():
	tvt = 'train'
	videos = os.listdir('./features/' + tvt +'/')
	video = videos[0]
	frames = os.listdir('./features/' + tvt + '/' + video)
	frame = np.load('./features/' + tvt +'/'+ video + '/' 
	                        + frames[0])

	num=0
	f, ax = plt.subplots(2,2)
	ax[0,0].imshow(frame[:,:,num], cmap = 'gray')
	ax[0,1].imshow(frame[:,:,num+1], cmap = 'gray')
	ax[1,0].imshow(frame[:,:,num+2], cmap = 'gray')
	ax[1,1].imshow(frame[:,:,num+3], cmap = 'gray')

	plt.show()

def is_prime(num):
	if num >1:
		for i in range(2, int(num**0.5)):
			if (num%i)==0:
				print(num, "is not a prime number:", i,"times", num/i)
				break
		else:
			print(num, "is a prime number")
	else:
		print(num, "is a prime number")

def mean_cal(tvt):
    videos = os.listdir('./images/' + tvt +'/')
    r,g,b = [], [], []
    print 'total number of videos {}'.format(len(videos))
    for video in videos:
        frames = os.listdir('./images/' + tvt + '/' + video)
        frames = sorted(frames)
        for i in frames:
            frame = plt.imread('./images/' + tvt +'/'+ video + '/' + i)
            r.append(np.mean(frame[:,:,0]))
            g.append(np.mean(frame[:,:,1]))
            b.append(np.mean(frame[:,:,2]))
    f = lambda x: sum(x)/len(x)
    res = [f(r), f(g), f(b)]
    print("Mean value for "+tvt+" is R: {}, G: {}, B: {}".format(
    res[0],res[1],res[2]))
    return res 

def videoReader(filename, root_directory):
    
    video_filename = os.path.join(root_directory,filename)
    capture = cv2.VideoCapture()
    success = capture.open(video_filename)
    
    if not success:
        print "Couldn't open video"
        print "Crashed at index %d." % index
        return None
        
    if width < 0:
        width = capture.get(CV_CAP_PROP_FRAME_WIDTH)
    if height < 0:
        height = capture.get(CV_CAP_PROP_FRAME_HEIGHT)
