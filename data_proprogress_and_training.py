import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random
from data_process_utils import get_border_need_add, read_data
from cnn_training import cnn_training

my_face_path = './my_face/'
other_face_path = './other_face/'

img_size = 64

imgs = []
labels = []

my_face_imgs, my_face_labels = read_data(my_face_path, img_size)
other_face_imgs, other_face_label = read_data(other_face_path, img_size)
imgs[len(imgs):len(my_face_imgs)] = my_face_imgs
imgs[len(my_face_imgs):len(other_face_imgs)] = other_face_imgs
labels[len(labels):len(my_face_labels)] = my_face_labels
labels[len(my_face_labels):len(other_face_label)] = other_face_label
#process labels and image as ndarry format
imgs = np.array(imgs)
'''remember the vector that the [1  0] as me and [0  1] may be other people's face'''
labels = np.array([[1, 0] if lab == my_face_path else [0, 1] for lab in labels])

#dataset split and use random split the dataset
train_x, test_x, train_y, test_y = train_test_split(imgs, labels, test_size=0.05,
                                                    random_state=random.randint(0, 100))
#dataset reshape  #x.shape[0] means the dataset's row#
train_x = train_x.reshape(train_x.shape[0], img_size, img_size, 3)
test_x = test_x.reshape(test_x.shape[0], img_size, img_size, 3)
'''dataset's data type convert to ...'''
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0
print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

'''get the image'''
batch_size = 10
num_batch = len(train_x)//batch_size

x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob = tf.placeholder(tf.float32)
keep_prob_fully = tf.placeholder(tf.float32)

# if __name__ == "__main__":
cnn_training(x, keep_prob, keep_prob_fully, y_, num_batch, batch_size, train_x, train_y, test_x, test_y)


























































































