from datasets.augLib import class_map
from datasets.dataset_utils import int64_feature, bytes_feature
import tensorflow as tf

from PIL import Image
import cv2
import numpy as np
import os

DIRECTORY_ROOT = 'F:/data/augLib/'

DIRECTORY_TRAIN = 'train'
DIRECTORY_TEST = 'test'


def _process_one_image(dir_root,directory, name):
    """
    :param dir_root: the path of train or test dataset
    :param directory: the name of dir where all the images have the same class
    :param name: the filename of image
    :return:
        image_data, notice! encoder and decoder
        label, is not one_hot label
    """

    if name.split('.')[-1] not in ['jpg','jpeg','png','bmp']:
        raise ValueError(name,'is not the image format')
    if directory not in class_map:
        raise ValueError('no class:', directory)
    label = class_map[directory]
    filename = dir_root+'/'+directory + '/' + name
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename,'not found')

    # image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # im = Image.open(filename)
    # img_data = im.tobytes()  # 原生raw bytes

    img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)  # BGR,<class 'numpy.ndarray'>
    if img is None:
        raise ValueError('read s% failed' % filename)
    img_data = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    # data normalization
    # img_resize = img_resize / np.max(img_resize)
    # img_resize = img_resize / 255.

    # img_resize = img_resize - np.mean(img_resize)
    # img_resize = img_resize.reshape(-1)

    return img_data.tobytes(), label


def _convert_to_example(img_data,label):
    example = tf.train.Example(features = tf.train.Features(feature={
        'image/data': bytes_feature(img_data),
        'image/label': int64_feature(label)
    }))
    return example


def run(dataset_dir, output_dir, name, shuffling=False):
    """
    :param dataset_dir: the path of train or test dataset
    :param output_dir:
    :param name: the name of the tfrecord
    :param shuffling:
    :return:
    """
    tf_filename = output_dir+'/'+ name + '.tfrecord'
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for class_name in os.listdir(dataset_dir):
            print('handle directory:',class_name)
            class_path = dataset_dir+'/'+class_name
            for filename in os.listdir(class_path):
                print('>>>convert image:',filename)
                img_data,label = _process_one_image(dataset_dir,class_name,filename)
                example = _convert_to_example(img_data,label)
                tfrecord_writer.write(example.SerializeToString())




def _test_process_one_image():
    img, label = _process_one_image(DIRECTORY_ROOT + DIRECTORY_TRAIN, 'Caterpillar', '2_aug_1.bmp')
    print(type(img))  # <class 'numpy.ndarray'>
    print(img.shape)  # (224, 224, 3)  灰度图怎么也是3通道
    # IMREAD_UNCHANGED 读取为(224,224)形状的图像，没有第三维度需要reshape
    print(img.dtype)  # uint8
    print(label)


def _test_run():
    run(DIRECTORY_ROOT+DIRECTORY_TEST,DIRECTORY_ROOT,'auglib_test')


if __name__ == '__main__':
    _test_run()

