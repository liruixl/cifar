import numpy as np
import os
import tensorflow as tf

# This implements DenseNet-40,L=40 k=12
# i.e. 3 dense blocks of 12 layers each.
# The growth rate is set to 12.
# 无 B C结构
# 数据集：CIFAR10

DENCE_BLOCK_NUM = 3
GROWTH_RATE = 12

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    if b'data' in dict:
        dict[b'data'] = dict[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1,32 * 32 * 3) / 256.
    return dict


def load_data_batch_one(path):
    batch = unpickle(path)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s: %d" % (path, len(data)))
    return data, labels


def load_data_batch_all(files, data_dir, label_count):
    data, labels = load_data_batch_one(data_dir+'/'+files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_batch_one(data_dir+'/'+files[0])
        data = np.append(data,data_n,axis=0)
        labels = np.append(labels,labels_n,axis=0)  # list
    labels = np.array([ [ float(i == label) for i in range(label_count) ] for label in labels ]) # one hot编码
    return data,labels

# ???
def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):
    res = [ 0 ] * len(tensors)  # [0, 0]
    batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]
    total_size = len(batch_tensors[0][1]) # data[test] 长度
    batch_count = (total_size + batch_size - 1) // batch_size  # 取上整
    for batch_idx in range(batch_count):
        current_batch_size = None
        for (placeholder, tensor) in batch_tensors:
            batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]  # 切片如果越界，则只切到末尾
            current_batch_size = len(batch_tensor)  # 最后一个不一定等于batch_size
            feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]
        tmp = session.run(tensors, feed_dict=feed_dict)  # [ cross_entropy, accuracy ]
        res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]  # 每个batch的损失及准确率的叠加
    return [ r / float(total_size) for r in res ]  # 平均一下



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


def run_model(data, image_dim, label_count, depth):
    weight_decay = 1e-4
    # 在pyhton3中，"/"表示的就是float除，不需要再引入模块，就算分子分母都是int，返回的也将是浮点数
    layers = (depth-4) // DENCE_BLOCK_NUM  # 论文中对应imagenet应该是减5
    graph = tf.Graph()

    # ==========================创建图================================
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, image_dim])
        ys = tf.placeholder("float", shape=[None, label_count])
        lr = tf.placeholder("float", shape=[])
        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder("bool", shape=[])

        current = tf.reshape(xs, [-1, 32, 32, 3])
        current = conv2d(current, 3, 16, 3)     #  先变成16通道的

        # BLOCK1
        current,features = block(current,layers,16,GROWTH_RATE,is_training,keep_prob)
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
        current = avg_pool(current, 8)  # 全局池化->(N,1,1,features)

        final_dim = features
        current = tf.reshape(current, [-1, final_dim])  # Flatten
        # FC layer
        Wfc = weight_variable([final_dim, label_count])
        bfc = bias_variable([label_count])
        ys_ = tf.nn.softmax(tf.matmul(current, Wfc) + bfc)  # 矩阵相乘

        cross_entropy = -tf.reduce_mean(ys * tf.log(ys_ + 1e-12))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
        correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # =========================结束创建图=====================================

    with tf.Session(graph=graph) as sess:
        batch_size = 64
        learning_rate = 0.1
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        # ==================================数据部分start============================
        train_data, train_labels = data['train_data'], data['train_labels']
        batch_count = len(train_data) // batch_size
        # 丢弃batch_count * batch_size后面不足一个batch_size的train_data
        batches_data = np.split(train_data[:batch_count * batch_size], batch_count)
        batches_labels = np.split(train_labels[:batch_count * batch_size], batch_count)
        print ("Batch per epoch: ", batch_count)

        # ====================================end====================================

        for epoch in range(1, 1 + 300):
            if epoch == 150: learning_rate = 0.01
            if epoch == 225: learning_rate = 0.001
            for batch_idx in range(batch_count):
                xs_, ys_ = batches_data[batch_idx], batches_labels[batch_idx]
                batch_res = sess.run([train_step, cross_entropy, accuracy],
                                        feed_dict={xs: xs_, ys: ys_, lr: learning_rate, is_training: True,
                                                   keep_prob: 0.8})
                if batch_idx % 100 == 0:
                    print(epoch, batch_idx, batch_res[1:])

            # if  not os.path.isdir('densenet_%d.ckpt' % epoch):
            #     os.mkdir('densenet_%d.ckpt' % epoch)
            save_path = saver.save(sess, './checkpoint_dir/densenet_%d.ckpt' % epoch,)
            test_results = run_in_batch_avg(sess, [cross_entropy, accuracy], [xs, ys],
                                            feed_dict={xs: data['test_data'], ys: data['test_labels'],
                                                       is_training: False, keep_prob: 1.})
            print(epoch, batch_res[1:], test_results)


def run():
    data_dir = r'F:\data\cifar-10-batches-py'
    image_size = 32
    image_dim = image_size * image_size * 3
    meta = unpickle(data_dir + '/batches.meta')
    label_names = meta[b'label_names']
    label_count = len(label_names)
    print([str(byte,encoding='utf-8') for byte in label_names],label_count)

    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data_batch_all(train_files, data_dir, label_count)
    pi = np.random.permutation(len(train_data))
    train_data, train_labels = train_data[pi], train_labels[pi]
    test_data, test_labels = load_data_batch_all(['test_batch'], data_dir, label_count)

    print("数据加载完成：")
    print ("Train:", np.shape(train_data), np.shape(train_labels))
    print ("Test:", np.shape(test_data), np.shape(test_labels))

    data = {'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels}

    run_model(data, image_dim, label_count, 40)


if __name__ == '__main__':
    run()