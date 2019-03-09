import tensorflow as tf
from myface_net import cnn_layer
import sys

def cnn_training(x, keep_prob, keep_prob_fully, y_, num_batch, batch_size, train_x, train_y, test_x, test_y):
    out = cnn_layer(x, keep_prob=keep_prob, keep_prob_fully=keep_prob_fully)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy_loss)
    #accuracy compute
    accuracy = tf.reduce_mean(tf.cast((tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))), tf.float32))
    '''save the loss and accracy then it will be convenient to tensorboard'''
    tf.summary.scalar('loss', cross_entropy_loss)
    tf.summary.scalar('accracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    #data saver initial
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./tmp/', graph=tf.get_default_graph())
        flag = 0
        for n in range(10):
            for i in range(num_batch):
                batch_x = train_x[i : (i+1)*batch_size]
                batch_y = train_y[i : (i+1)*batch_size]
                '''taining'''
                _, loss, summary = sess.run([train_step, cross_entropy_loss, merged_summary_op],
                                            feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5, keep_prob_fully:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                '''print loss'''
                print(n*num_batch+i, loss)
                if(n*num_batch+i)%100 == 0:
                    acc =  accuracy.eval({x:test_x, y_:test_y, keep_prob:1.0, keep_prob_fully:1.0})
                    print(n*num_batch+i, acc)
                    # if acc>0.98 and n>2:
                    if acc > 0.96 and flag!=0:
                        saver.save(sess, './train_model/train_face.model', global_step=n*num_batch+i)
                        sys.exit(0)
                    flag = flag+1
        print("The accuracy is less 0.98")


































