import tensorflow as tf
import cv2
import dlib
from myface_net import cnn_layer

img_size = 64
x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
keep_prob = tf.placeholder(tf.float32)
keep_prob_fully = tf.placeholder(tf.float32)

out_res = cnn_layer(x, keep_prob, keep_prob_fully)
'''
out_res: [[ 0.45090234 -1.5836945 ]]
out_res: [[ 0.45283878 -1.5856509 ]]
out_res: [[ 0.4533069 -1.5893035]]
these result means which our class is two. And the result should out the 
vector [0   1] or [1   0] that the above result is about the two vectors.
'''
'''
tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，
如果vector是一个向量，那就返回一个值，如果是一个矩阵，
那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
'''
predict = tf.argmax(out_res, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('./train_model/.'))

def whose_face(face_image):
    res = sess.run(predict, feed_dict={x: [face_image/255.0], keep_prob:1.0, keep_prob_fully:1.0})
    '''
    res: [0]
    res: [0]
    res: [0]
    res: [0]
    '''
    # print("res:", res)
    # print("out_res:", sess.run(out_res, feed_dict={x: [face_image/255.0], keep_prob:1.0, keep_prob_fully:1.0}))
    # print("predict:", predict.eval(sess))
    '''res contains the result of the net, which is the position(or index)
    of the max value that is the [1 0] or [0 1] has the one's position'''
    if res[0] == 0:
        return True
    else:
        return False

cap = cv2.VideoCapture(0)
facedetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
face_detector = dlib.get_frontal_face_detector()
x0 = 0
y0 = 0
x1 = 0
y1 = 0

while(True):
    ret, frame = cap.read()
    # grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame, 1)
    for i, position in enumerate(faces):
        pos = position
        x0 = pos.left()
        y0 = pos.top()
        x1 = pos.right()
        y1 = pos.bottom()
    if 0!=x0 and 0!=y0 and 0!=x1 and 0!=y1:
        face_img = frame[y0:y1, x0:x1]
        resize_face_img = cv2.resize(face_img, (img_size, img_size))
        result = whose_face(resize_face_img)
        if result:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, "king", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.imshow('wind', frame)
        else:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, "other", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.imshow('wind', frame)
    else:
        cv2.imshow("wind", frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



































