import cv2
import dlib
import numpy as np
import os

other_face_path = './lfw/'
file_dir_names = []
file_names = []
file_paths = []
x0 = 0
y0 = 0
x1 = 0
y1 = 0
for file_dir in os.listdir(other_face_path):
    file_dir_names.append(file_dir)
for file_dir_name in file_dir_names:
    file_path = other_face_path+file_dir_name
    for file in os.listdir(file_path):
        file_names.append(file)
        file_paths.append(file_path+"/"+file)
print(file_paths)
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
j = 0
for face_img in file_paths:
    img = cv2.imread(face_img)
    dets = detector(img, 1)
    img_name = face_img.split("/")[-1]
    for i, position in enumerate(dets):
        pos = position.rect
        x0 = pos.left()
        y0 = pos.top()
        x1 = pos.right()
        y1 = pos.bottom()
        if 0!=x0 and 0!=x1 and 0!=y0 and 0!=y1:
            print("Processing %d", j)
            j = j+1
            tem_img = img[y0:y1, x0:x1]
            cv2.imwrite('./other_face/{}'.format(img_name), tem_img)
        else:
            pass



















