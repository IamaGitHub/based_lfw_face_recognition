import cv2
import os

img_path = './other_face/'
for file in os.listdir(img_path):
    path = img_path+file
    img = cv2.imread(path)
    if img is None:
        print("file is :------", path)
        os.remove(path)