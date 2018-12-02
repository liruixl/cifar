import numpy as np
import tensorflow as tf
import os
import cv2

class_map = {
    'Caterpillar': 0,
    'Dirty': 1,
    'Flat flower': 2,
    'Hole': 3,
    'Mold': 4,
    'pressing': 5,
    'Scratch': 6,
    'Spot': 7,
    'Warped': 8,
    'Zinc ash': 9,
    'Zinc residue': 10,
}

class_num = 11

root_dir = r'F:\data\augLib'
train_dir = r'F:\data\augLib\train'
test_dir = r'F:\data\augLib\test'


def _get_filelist(dir_path):
    file_list = []
    if not os.path.exists(dir_path):
        raise FileNotFoundError('dir_path is not exist!')
    for name in os.listdir(dir_path):
        file_list.append(os.path.join(dir_path,name))
    return file_list


def get_file(data_dir):
    # data_dir : train or test directory
    img_list = []
    label_list = []
    for class_dir in _get_filelist(data_dir):
        if os.path.isdir(class_dir):
            # 注意，这里是windows下路径，最好统一成"/"
            class_dir = class_dir.replace('\\','/')
            class_name = class_dir.split('/')[-1]
            if class_name not in class_map:
                raise ValueError('class_map have no %s class' % class_name)
            filelist = _get_filelist(class_dir)
            # print(class_dir,len(filelist))
            img_list = img_list + filelist
            label_list = label_list \
                                + [class_map[class_name]]*len(filelist)
    onehot_label_list = [[float(i==label) for i in range(class_num)]for label in label_list]
    return img_list,onehot_label_list


def get_nextBatch(image, label, batch_size, step):
    start = batch_size * step
    end = batch_size * (step + 1)
    image_batch = image[start: end]
    label_batch = label[start: end]
    data_batch = []
    for img in image_batch:
        # print(img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        img_resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # data normalization
        img_resize = img_resize / np.max(img_resize)
        img_resize = img_resize - np.mean(img_resize)
        img_resize = img_resize.reshape(-1)

        data_batch.append(img_resize)

    data_batch = np.array(data_batch)
    label_batch = np.array(label_batch)

    return np.array(data_batch), np.array(label_batch)


def read_tfrecord(tf_filename,shape):
    """
    :param tf_filename: tfrecord name or tfrecords list or string Tensor,shape = [?]?
    :param shape: decode img_data to the shape
    :return: Tensor,img and label
    """
    if not isinstance(tf_filename,list) and not isinstance(tf_filename,tf.Tensor):
        tf_filename = [tf_filename]
    tf_filename_queue = tf.train.string_input_producer(tf_filename)
    tf_reader = tf.TFRecordReader()
    _, serialized_example = tf_reader.read(tf_filename_queue)
    img_tensor,label_tensor = _parse_example(serialized_example,shape)

    return img_tensor,label_tensor


def _parse_example(serialized_example,shape):
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/data': tf.FixedLenFeature([], tf.string),
                                           'image/label': tf.FixedLenFeature([], tf.int64),
                                       })
    img = tf.decode_raw(features['image/data'], tf.uint8) # 解析时必须指定类型，不同类型所占字节数不一样
    img = tf.reshape(img,shape=shape)  # uint8

    # img = img/255.  # 不支持除法
    img = tf.cast(img, tf.float32) * (1. / 255)-0.5
    img = img - tf.reduce_mean(img)

    label = features['image/label']  # dtype=int64   int64_list...
    label = tf.cast(label, tf.int32)

    return img, label


def _test_read_tfrecord():
    import matplotlib.pyplot  as plt

    # tf_name = tf.constant([r'F:\data\augLib\auglib_test.tfrecord', r'F:\data\augLib\auglib_test.tfrecord'],dtype = tf.string)
    tf_name = [r'F:\data\augLib\auglib_test.tfrecord', r'F:\data\augLib\auglib_test.tfrecord']
    img, label = read_tfrecord(tf_name, [224, 224])
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)
          # 不要忘记这句

        # 队列线程thread 的开始，跟 feed_dict 有冲突？
        # 开始时 queue 输入还是个placeholder
        im  = sess.run(img)
        print(type(im))
        plt.imshow(im,cmap='gray') # raise TypeError("Invalid dimensions for image data") [224,224] or [224,224,3/4]
        plt.show()

        im2 = sess.run(img)
        print(type(im2))
        plt.imshow(im2)  # raise TypeError("Invalid dimensions for image data") [224,224] or [224,224,3/4]
        plt.show()

        coord.request_stop()
        coord.join(thread)


if __name__ == '__main__':
    _test_read_tfrecord()


