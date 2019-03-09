import numpy as np
import cv2
import os

'''get the border that we needed'''
def get_border_need_add(img_path):
    img = cv2.imread(img_path)
    print("name:", img_path)
    h, w, _ = img.shape
    largest = max(h, w)
    left, top, right, bottom = (0, 0, 0, 0)

    if largest > w:
        left = (largest-w)//2
        right = w - left
        return left, top, right, bottom
    elif largest > h:
        top = (largest-h)//2
        bottom = h - top
        return left, top, right, bottom
    else:
        pass
    return left, top, right, bottom

'''process the data as a img_set and label_set which return two list'''
def read_data(img_path, img_size):
    img_data = []
    img_label = []
    for file in os.listdir(img_path):
        filename = file
        img = cv2.imread(img_path+filename)
        left, top, right, bottom = get_border_need_add(img_path+filename)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
        img = cv2.resize(img, (img_size, img_size))
        img_data.append(img)
        img_label.append(img_path)
    return img_data, img_label
























