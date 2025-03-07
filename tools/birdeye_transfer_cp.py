import cv2
import math
import numpy as np

img = cv2.imread('C:/Users/CSY/SMILEtrack-main/BoT-SORT/datasets/sports/test/img1/000001.png')

camera_parameter = []
f = open('C:\\Users\\CSY\\SMILEtrack-main\\BoT-SORT\\datasets\\sports\\re0013.txt', 'r')
for line in f.readlines():
    camera_parameter.append(eval(line))
f.close


intrinsic_matrix, extrinsic_matrix = camera_parameter[0], camera_parameter[1]
rotation_matrix = extrinsic_matrix[:, :3]
translation_matrix = extrinsic_matrix[:, 3]

bird_rotation_matrix = 1
bird_translation_matrix = 2