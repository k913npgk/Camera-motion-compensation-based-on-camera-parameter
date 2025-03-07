import cv2
import numpy as np


f = open('tracker/test/gt/gt.txt', mode='r')

line = f.readline()
now_frame = 0
while line:
    data = line.split(", ")
    if data[0] != str(now_frame):
        if int(data[0]) < 10:
            img = cv2.imread('tracker/test/img1/00000'+str(data[0])+'.jpg')
        elif int(data[0]) < 100:
            img = cv2.imread('tracker/test/img1/0000'+str(data[0])+'.jpg')
        else:
            img = cv2.imread('tracker/test/img1/000'+str(data[0])+'.jpg')
        img_height, img_width,_ = img.shape
        now_frame = int(data[0])
        
    #yolo
    # x_center = float(data[2]) * img_width
    # y_center = float(data[3]) * img_height
    # W = float(data[4]) * img_width
    # H = float(data[5]) * img_height
    # xmin  = x_center - W/2
    # ymin  = y_center - H/2
    
    xmin = float(data[2])
    ymin = float(data[3])
    W = float(data[4])
    H = float(data[5])
    
    crop_img = img[int(ymin):int(ymin+H), int(xmin):int(xmin+W)]
    #cv2.imwrite('trans_img/frame'+str(int(data[0])%2)+'/'+data[0]+'-'+str(data[1])+'.jpg', crop_img)
    cv2.imwrite('tracker/trans_img/P'+data[1]+'/'+data[0]+'.jpg', crop_img)
    
    line = f.readline()