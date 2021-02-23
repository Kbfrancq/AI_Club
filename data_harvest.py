# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:28:33 2021
@author: Conley Price
Description: This script will help our nueral network get better data to train on. 
We are going to get a email sent out through the school of IT 
to send 3-4 second videos of themselves with and without a mask. This script will make those videos to images.
"""

import cv2
import os

#Folder path with .mp4 video files
mask_mp4_path = "mask_videos"
non_mp4_path = "non_videos"

#Folder path with jpg files will be sent
mask_jpg_path = "mask_JPG"
non_jpg_path = "non_JPG"

videos = 0
pictures = 0

#saves the frame of video as jpg in the folderPath folder
def getFrame(vid,folderPath,filename,sec,count):
    vid.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vid.read()
    if hasFrames:
        cv2.imwrite(folderPath+"\\"+filename+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

def convert(fp,fileName,end,videos,pictures):
    sec = 0
    frameRate = 0.5       #Make number number lower to capture more frames, right not it is at one frame every half a second
    count=1
    pictures = pictures + 1    #Keep track of how many pictures, to display at end
    vidcap = cv2.VideoCapture(fp+fileName)
    print(fileName)
    videos = videos +1
    success = getFrame(vidcap,end,fileName,sec,count)
    while success:
        count = count + 1
        pictures = pictures + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(vidcap,end,fileName,sec,count)
    return videos, pictures
            
#runs the conversion for all mp4 files in the non_mp4_path folder
for filename in os.listdir(non_mp4_path):
    videos, pictures = convert(non_mp4_path + "\\",filename, non_jpg_path,videos,pictures)
#runs the conversion for all mp4 files in the mask_mp4_path folder#runs the conversion for all mp4 files in the non_mp4_path folder
for filename in os.listdir(mask_mp4_path):
    videos, pictures = convert(mask_mp4_path + "\\",filename, mask_jpg_path,videos,pictures)
        
#console output done when done
print("Converted ", videos, " videos to ", pictures, " pictures.")
print("The images with mask are in ", mask_mp4_path, " and the images without mask are in ", non_mp4_path, " folder." )
