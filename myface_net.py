import tensorflow as tf
from cnn_utils import weightVariable, biasVariable,dropout,conv2d, maxPool

def cnn_layer(x, keep_prob, keep_prob_fully):
    #first layer
    '''
    filter用于指定CNN中的卷积核，它要求是一个Tensor，
    具有[filter_height, filter_width, in_channels, out_channels]
    这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    ，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
    ，这里是维度一致，不是数值一致。这里out_channels指定的是卷积核的个数，
    而in_channels说明卷积核的维度与图像的维度一致，在做卷积的时候，单个卷积核在不同维度上对应的卷积图片，
    然后将in_channels个通道上的结果相加，加上bias来得到单个卷积核卷积图片的结果。
    '''
    W1 = weightVariable([3, 3, 3, 32])
    b1 = biasVariable([32])
    #cnn
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    #pool
    pool1 = maxPool(conv1)
    #droupout
    drop1 = dropout(pool1, keep_prob)

    #second layer
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    #cnn
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    #pool
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob)

    #third layer
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    #cnn
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    #pool
    pool3 = maxPool(conv3)
    #drop
    drop3 = dropout(pool3, keep_prob)

    #fully connect layer
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flatten = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flatten, Wf) + bf)
    dropf = dropout(dense, keep_prob_fully)

    #out layer
    Wout = weightVariable([512, 2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out
'''cnn training function'''
# def cnn_training(x, keep_prob, keep_prob_fully):
#     cnn_layer(x, keep_prob, keep_prob_fully)

















