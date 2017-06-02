import os
import sys
import cv2
from subprocess import call

out_path = '/home/wenjing/video/images/'

def dump_frames(vid_path):
    vid_name = vid_path.split('/')[-1].split('.')[0]
    #print vid_name, vid_name[-6:-4]
    # g01-g08 test, g09-g20 train, g21-g25 val, according to list1
    if int(vid_name[-6:-4]) > 20:
        out_full_path = os.path.join(out_path, 'val/', vid_name)
    elif int(vid_name[-6:-4]) > 8:
        out_full_path = os.path.join(out_path, 'train/', vid_name)
    else:
        out_full_path = os.path.join(out_path, 'test/', vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    dest = out_full_path+'/%06d.jpg'
    call(["ffmpeg", "-i", vid_path, dest])
    print '{} done'.format(vid_name)


def check_convert():
    tvt=['train', 'val', 'test']

    for i in tvt:
        videos = os.listdir('./images/' + i +'/')
        for video in videos:
            images = os.listdir('./images/' + i +'/'+video+'/')
            path='./UCF-101/'+video[2:-8]+'/'+video+'.avi'
            v = cv2.VideoCapture(path) 
            fcount = int(v.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            assert abs(fcount-len(images))<2, str(fcount)+' '+str(len(images))
    print 'Checked, frame number difference less than 2'

#numpy.random.randint(10, size=2) two random class -> 9,85,17,93,47

def convert():
    num = [17, 93, 47]
    f = open("./data/ucfTrainTestlist/classInd.txt","r")
    videos = f.readlines()
    f.close
    videos = [video.strip() for video in videos]
    for i in num:
        temp = videos[i].split()
        listing = os.listdir('./UCF-101/'+temp[1]+'/')
        for j in listing:
            dump_frames('./UCF-101/'+temp[1]+'/'+j)
    


