import numpy as np
import os
import tensorflow as tf

"""
Bottleneck layers.
    BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
In our experiments, we let each 1×1 convolution produce 4k feature-maps
Compression.(transition layers)
If a dense block contains m feature-maps, we let
the following transition layer generate floor[θm] output featuremaps, 
we set θ=0.5 in our experiment

Note that each “conv” layer shown in the
table corresponds the sequence BN-ReLU-Conv

DenseNet architectures for ImageNet. 
The growth rate for all the networks is k = 32.

In our experiments on ImageNet, we use a DenseNet-BC
structure with 4 dense blocks on 224×224 input images.
The initial convolution layer comprises 2k convolutions of
size 7×7 with stride 2
"""


class DenseNet_BC(object):

    def __init__(self, input, num_classes,k,layers):
        self.input = input
        self.num_classes = num_classes
        self.growth_rate = k
        self.layers = layers
        self.build_graph()

    def build_graph(self):
        # layers = (self.depth - 4) // self.block_num
        layers = self.layers
        is_training = True
        keep_prob = 0.75
        GROWTH_RATE = self.growth_rate
        # ===========================注意 卷积核的通道数 ，数据集不同时需要修改=======================
        current = self.input
        print('input shape:',self.input.get_shape())
        current = conv2d(current, 1, 2*GROWTH_RATE, 3)  # 2k 通道
        current = tf.nn.max_pool(current,[1,3,3,1],[1,2,2,1],padding='SAME')

        # BLOCK1
        current, features = block_B(current, layers[0], 2*GROWTH_RATE, GROWTH_RATE, is_training, keep_prob,is_B=True)
        # Transition layers 1
        current = batch_activ_conv(current, features, features//2, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        # BLOCK2
        current, features = block_B(current, layers[1], features//2, GROWTH_RATE, is_training, keep_prob,is_B=True)
        # Transition layers 2
        current = batch_activ_conv(current, features, features//2, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        # Block 3
        current, features = block_B(current, layers[2], features//2, GROWTH_RATE, is_training, keep_prob,is_B=True)
        # Transition layers 3
        current = batch_activ_conv(current, features, features // 2, 1, is_training, keep_prob)
        current = avg_pool(current, 2)
        # Block 4
        current, features = block_B(current, layers[3], features//2, GROWTH_RATE, is_training, keep_prob, is_B=True)
        # a global average pooling, 有BN层和ReLU层？
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current)
        # =================================注意！！====这里的也跟数据集图像尺寸有关==================
        current = avg_pool(current, 9)  # 全局池化->(N,1,1,features)

        final_dim = features
        current = tf.reshape(current, [-1, final_dim])  # Flatten
        # FC layer
        Wfc = weight_variable([final_dim, self.num_classes])
        bfc = bias_variable([self.num_classes])
        self.logits = tf.nn.softmax(tf.matmul(current, Wfc) + bfc)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(input, in_features, out_features, kernel_size, with_bias=True):
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


def block_B(input,layers,in_features,growth,is_training,keep_prob,is_B = False):
    # Bottleneck layers.
    current = input
    features = in_features
    for i in range(layers):
        if is_B:
            temp = batch_activ_conv(current,features,4*growth,1,is_training,keep_prob)
            temp = batch_activ_conv(temp, 4*growth, growth, 3, is_training, keep_prob)
        else:
            temp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
        current = tf.concat((current, temp), axis=3)
        features = features + growth
    return current, features


def avg_pool(input, s):
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')