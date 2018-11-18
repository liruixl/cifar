import tensorflow as tf
import numpy as np
from datasets.Data_Augmentation_NEU import get_file
from datasets.Data_Augmentation_NEU import get_nextBatch

class AlexNet(object):

    def __init__(self, input, num_classes,keep_prob):
        self.input = input
        self.num_classes = num_classes
        self.KEEP_PROB = keep_prob
        self.build_graph()

    def build_graph(self):
        conv1 = conv2d(self.input,11,96,4,name='conv1')
        norm1 = lrn(conv1,name='norm1')
        self.pool1 = max_pooling(norm1,3,2,name='pooling1',padding='SAME')

        conv2 = conv2d(self.pool1,5, 256,1,'conv2')
        norm2 = lrn(conv2,name='norm2')
        self.pool2 = max_pooling(norm2,3,2,name='pooling2',padding='SAME')

        conv3 = conv2d(self.pool2, 3, 384, 1, name='conv3')
        conv4 = conv2d(conv3, 3, 384, 1, name='conv4')

        conv5 = conv2d(conv4, 3, 256, 1, name='conv5')
        self.pool5 = max_pooling(conv5,3,2,name='pooling5',padding='SAME')

        flattened = tf.reshape(self.pool5,[-1,5*5*256])

        fc6 = fc(flattened,5*5*256,4096,'fc6')
        droupt6 = dropout(fc6, self.KEEP_PROB)
        fc7 = fc(droupt6,4096,4096,'fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        self.fc8 = fc(dropout7,4096,self.num_classes,'fc8',is_relu=False)



def weight_variable(shape):
    weight = tf.get_variable('weights',shape=shape,
                             initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))
    return weight


def bias_variable(shape):
    bias = tf.get_variable('bias',shape=shape,
                           initializer=tf.constant_initializer(value=0.01))
    return bias


def conv2d(input,ksize,num_filters,strides,name,padding='SAME'):
    in_channel = int(input.get_shape()[-1])

    with tf.variable_scope(name) as scope:
        w = weight_variable([ksize,ksize,in_channel,num_filters])
        b = bias_variable([num_filters])

        conv = tf.nn.conv2d(input,w,[1,strides,strides,1],padding=padding)
    return relu(conv)


def relu(input):
    return tf.nn.relu(input)


def max_pooling(input, ksize, strides, name, padding = 'SAME'):
    return tf.nn.max_pool(input,[1,ksize,ksize,1],[1,strides,strides,1],
                          padding=padding,name=name)


def lrn(x, name,radius=2, alpha=1e-4, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

def fc(input,num_in,num_out,name,is_relu=True):
    with tf.variable_scope(name) as scope:
        w = weight_variable([num_in,num_out])
        b = bias_variable([num_out])
    result = tf.matmul(input,w)+b
    if is_relu:
        return relu(result)
    else:
        return result


if __name__ == '__main__':

    # input = tf.constant(1.0, shape=[1, 144, 144, 1])
    # model = AlexNet(input, 10, 0.5)
    # print(model.pool5)

    IMG_HEIGHT = 144
    IMG_WIDTH = 144
    IMG_DEPTH = 1

    num_class = 6
    batch_size = 16
    learning_rate = 0.00001

    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH])
        ys = tf.placeholder("float", shape=[None, num_class])

        model = AlexNet(xs,num_class,0.5)
        fc8 = model.fc8

        #`logits` and `labels` must have the same shape,
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys,
                                                                logits=fc8)
        loss = tf.reduce_mean(cross_entropy,name = 'loss')
        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9).minimize(loss=loss)
        correct_prediction = tf.equal(tf.argmax(fc8, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    TRAIN_DATA_PATH = 'F:\\data\\NEU\\train\\'
    TEST_DATA_PATH = 'F:\\data\\NEU\\test\\'
    train_data, train_labels = get_file(TRAIN_DATA_PATH)
    test_data, test_labels = get_file(TEST_DATA_PATH)

    batch_count = len(train_data) // batch_size
    print('batch_num:',batch_count)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        saver.restore(sess,'ckpt_alexnet_neu/ckpt_20')

        for epoch in range(1,1+15):
            if epoch == 10: learning_rate = 0.00001
            if epoch == 15: learning_rate = 0.00001
            for step in range(batch_count):
                train_data_, train_labels_ = get_nextBatch(train_data,
                                                           train_labels,
                                                           batch_size,
                                                           step)
                train_data_ = np.reshape(train_data_,[-1,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH])

                _,train_loss, acc = sess.run([train_op,loss,accuracy],
                                              feed_dict={xs:train_data_,
                                                         ys:train_labels_,})
                if step % 100 == 0:
                    print(epoch,step,train_loss,acc)

            print('测试第%d个epoch' % epoch)
            loss_acc = [0,0]
            test_batch_count = (len(test_data)+ batch_size-1)//batch_size
            for step in range(test_batch_count):
                test_data_, test_labels_ = get_nextBatch(test_data,
                                                         test_labels,
                                                         batch_size,
                                                         step)
                current_size = len(test_data_)
                test_data_ = np.reshape(test_data_,[-1,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH])
                tmp = sess.run([loss,accuracy],feed_dict={xs:test_data_,
                                                          ys:test_labels_})
                loss_acc = [r + t for(r,t) in zip(loss_acc, tmp)]
            loss_acc =  [r/test_batch_count for r in loss_acc]
            print('loss and acc:',loss_acc)


            saver.save(sess,'ckpt_alexnet_neu/ckpt_%d' % epoch)

