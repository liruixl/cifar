import numpy as np
import tensorflow as tf
from datasets.Data_Augmentation_NEU import get_file
from datasets.Data_Augmentation_NEU import get_nextBatch

# This implements DenseNet-40,L=40 k=12
# i.e. 3 dense blocks of 12 layers each.
# The growth rate is set to 12.
# 无 B C结构


class DenseNet(object):

    def __init__(self, input, num_classes,depth,block_num,k):
        self.input = input
        self.num_classes = num_classes
        self.depth = depth
        self.block_num = block_num        # 3
        # self.layers = layers_per_block    # 12
        self.growth_rate = k
        self.build_graph()

    def build_graph(self):
        layers = (self.depth - 4) // self.block_num
        is_training = True
        keep_prob = 0.75
        GROWTH_RATE = self.growth_rate
        # ===========================注意 reshape ，数据集不同时需要修改=======================
        current = tf.reshape(self.input, [-1, 144, 144, 1])
        current = conv2d(current, 1, 16, 3)  # 先变成16通道的

        # BLOCK1
        current, features = block(current, layers, 16, GROWTH_RATE, is_training, keep_prob)
        # Transition layers 1
        current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        # BLOCK2
        current, features = block(current, layers, features, GROWTH_RATE, is_training, keep_prob)
        # Transition layers 2
        current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        # Block 3
        current, features = block(current, layers, features, GROWTH_RATE, is_training, keep_prob)
        # a global average pooling, 有BN层和ReLU层？
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current)
        # =================================注意！！====这里的也跟数据集图像尺寸有关==================
        current = avg_pool(current, 36)  # 全局池化->(N,1,1,features)

        final_dim = features
        current = tf.reshape(current, [-1, final_dim])  # Flatten
        # FC layer
        Wfc = weight_variable([final_dim, self.num_classes])
        bfc = bias_variable([self.num_classes])
        self.ys_ = tf.nn.softmax(tf.matmul(current, Wfc) + bfc)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv


# BN-ReLU-Conv
def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = conv2d(current,in_features,out_features,kernel_size)
    current = tf.nn.dropout(current,keep_prob)
    return current

# 3×3的卷积
def block(input, layers, in_features, growth, is_training, keep_prob):
    current = input
    features = in_features
    for i in range(layers):
        temp = batch_activ_conv(current,features,growth,3,is_training,keep_prob)
        current = tf.concat((current,temp),axis=3)
        features = features + growth
    return current, features


def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')



if __name__ == '__main__':
    # input = tf.constant(1.0, shape=[1, 144, 144, 1])
    # model = AlexNet(input, 10, 0.5)
    # print(model.pool5)

    IMG_HEIGHT = 144
    IMG_WIDTH = 144
    IMG_DEPTH = 1

    num_class = 6
    batch_size = 6
    learning_rate = 0.01

    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        ys = tf.placeholder("float", shape=[None, num_class])

        model = DenseNet(xs,6,depth=40,block_num=3,k=12)
        output = model.ys_

        # `logits` and `labels` must have the same shape,
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys,
                                                                logits=output)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9).minimize(loss=loss)
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    TRAIN_DATA_PATH = 'F:\\data\\NEU\\train\\'
    TEST_DATA_PATH = 'F:\\data\\NEU\\test\\'
    train_data, train_labels = get_file(TRAIN_DATA_PATH)
    test_data, test_labels = get_file(TEST_DATA_PATH)

    batch_count = len(train_data) // batch_size
    print('batch_num:', batch_count)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        # saver.restore(sess, 'ckpt_alexnet_neu/ckpt_10')

        for epoch in range(1, 1 + 20):
            if epoch == 10: learning_rate = 0.001
            if epoch == 15: learning_rate = 0.0001
            for step in range(batch_count):
                train_data_, train_labels_ = get_nextBatch(train_data,
                                                           train_labels,
                                                           batch_size,
                                                           step)
                train_data_ = np.reshape(train_data_, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

                _, train_loss, acc = sess.run([train_op, loss, accuracy],
                                              feed_dict={xs: train_data_,
                                                         ys: train_labels_, })
                if step % 100 == 0:
                    print(epoch, step, train_loss, acc)

            print('测试第%d个epoch' % epoch)
            loss_acc = [0, 0]
            test_batch_count = (len(test_data) + batch_size - 1) // batch_size
            for step in range(test_batch_count):
                test_data_, test_labels_ = get_nextBatch(test_data,
                                                         test_labels,
                                                         batch_size,
                                                         step)
                current_size = len(test_data_)
                test_data_ = np.reshape(test_data_, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
                tmp = sess.run([loss, accuracy], feed_dict={xs: test_data_,
                                                            ys: test_labels_})
                loss_acc = [r + t for (r, t) in zip(loss_acc, tmp)]
            loss_acc = [r / test_batch_count for r in loss_acc]
            print('loss and acc:', loss_acc)

            # saver.save(sess, 'ckpt_alexnet_neu/ckpt_%d' % epoch)
            saver.save(sess, 'ckpt_densenet_neu/densenet_neu_%d' % epoch)