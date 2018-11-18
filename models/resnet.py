
# from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from models import resnet_v1
import tensorflow.contrib.slim as slim
import tensorflow as tf


class ResNet50(object):
    def __init__(self,input,num_classes):
        self.input = input
        self.num_classes = num_classes
        self.build_graph()

    def build_graph(self):
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=1e-5)):
            logits, end_point = resnet_v1.resnet_v1_50(self.input, num_classes=6, scope='resnet_v1_50')
            # logits [-1,1,1,dim]  全局池化
            dim = logits.get_shape()[-1]
            self.logits = tf.reshape(logits,[-1,dim])






