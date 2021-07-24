#!/usr/bin/env python
import os
from moviepy.editor import *
import numpy as np

def extract_frames(movie, times, imgdir):
    clip = VideoFileClip(movie)
    for t in times:
        #imgpath = os.path.join(imgdir, '{}.jpg'.format(t)) # png
        imgpath = os.path.join(imgdir, 'frame_%06dms.jpg'%(1000*t)) # png
        print("save_frame to", imgpath);
        clip.save_frame(imgpath, t)

#t = 0.040*np.array([0, 104, 121, 139]); # 25fps
#t = 0.040*np.arange(104, 112); # 25fps
t = [3.320, 4.91, 5.78]; # 4.280, 
print("extract_frames to test_images_mixed/");
extract_frames('test_videos_output/challenge.mp4', t, 'test_images_mixed/');
