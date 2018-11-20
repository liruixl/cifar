from models.densenet_bc import DenseNet_BC
from models.resnet import ResNet50

from datasets import BatchFactory
import tensorflow as tf
import numpy as np
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_DEPTH = 1

    num_class = 11
    batch_size = 16
    learning_rate = 0.01

    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        ys = tf.placeholder("float", shape=[None, num_class])

        model = ResNet50(xs,num_class)
        # model = DenseNet_BC(xs,num_class,k=32,layers=[6,12,24,16])
        output = model.logits

        # `logits` and `labels` must have the same shape,
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys,
                                                                logits=output)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        # ==================注意：======不同的网络有不同的训练方法===================
        # densenetBC:we use a weight decay of 10−4 and a Nesterov momentum  of 0.9 without dampening.
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,momentum=0.9,
            use_nesterov=True).minimize(loss=loss + l2 * 1e-4)
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # ============================注意=====数据集，路径==============================
    dataset_name = 'augLib'
    TRAIN_DATA_PATH = r'F:/data/augLib/train'
    TEST_DATA_PATH = r'F:/data/augLib/test'
    ckpt_dir = 'ckpt/lib_resnet50'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    train_data, train_labels = BatchFactory.get_file(dataset_name,TRAIN_DATA_PATH,is_shuffle=True)
    test_data, test_labels = BatchFactory.get_file(dataset_name,TEST_DATA_PATH)

    batch_count = len(train_data) // batch_size
    print('batch_num:', batch_count)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_path and tf.train.checkpoint_exists(ckpt_path):
            saver.restore(sess, ckpt_path)
            print('restore model from ',ckpt_path)

        print('测试载入模型')
        loss_acc = [0, 0]
        N = np.zeros([num_class, num_class])  # 二维矩阵
        test_batch_count = (len(train_data) + batch_size - 1) // batch_size
        for step in range(test_batch_count):
            test_data_, test_labels_ = BatchFactory.get_next_batch(dataset_name,
                                                                   train_data,
                                                                   train_labels,
                                                                   batch_size,
                                                                   step)
            # current_size = len(test_data_)
            test_data_ = np.reshape(test_data_, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

            result = sess.run([output, loss, accuracy], feed_dict={xs: test_data_,
                                                                   ys: test_labels_})

            loss_acc = [r + t for (r, t) in zip(loss_acc, result[1:])]
            # test_labels_ = test_labels_.astype('int64')
            # a = np.max(result[0],axis=1)
            # b = np.max(test_labels_,axis=1)
            # print(type(result[0])) # <class 'numpy.ndarray'>
            # print(result[0].shape) # (16, 11)

            logits =  result[0]
            for r in range(len(logits)):
                rigth = list(test_labels_[r, :]).index(1.)
                # pre = list(result[r,:]).index(max(list(result[r,:])))
                pre = np.argmax(logits[r, :])
                N[rigth][pre] = N[rigth][pre] + 1
        loss_acc = [r / test_batch_count for r in loss_acc]
        print('loss and acc:', loss_acc)
        print(N)

        import sys
        sys.exit()



        for epoch in range(1, 1 + 100):
            temp = list(zip(train_data, train_labels))
            random.shuffle(temp)
            train_data, train_labels = zip(*temp)
            train_data, train_labels = list(train_data),list(train_labels)

            if epoch == 50:
                learning_rate = learning_rate/10
            if epoch == 75:
                learning_rate = learning_rate/10
            for step in range(batch_count):
                train_data_, train_labels_ = BatchFactory.get_next_batch(dataset_name,
                                                                         train_data,
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
            N = np.zeros([num_class, num_class])  # 二维矩阵
            test_batch_count = (len(test_data) + batch_size - 1) // batch_size
            for step in range(test_batch_count):
                test_data_, test_labels_ = BatchFactory.get_next_batch(dataset_name,
                                                                       test_data,
                                                                       test_labels,
                                                                       batch_size,
                                                                       step)
                # current_size = len(test_data_)
                test_data_ = np.reshape(test_data_, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

                result = sess.run([output, loss, accuracy], feed_dict={xs: test_data_,
                                                            ys: test_labels_})

                loss_acc = [r + t for (r, t) in zip(loss_acc, result[1:])]

                logits = result[0]
                for r in range(len(logits)):
                    rigth = list(test_labels_[r,:]).index(1.)
                    # pre = list(result[r,:]).index(max(list(result[r,:])))
                    pre = np.argmax(logits[r,:])
                    N[rigth][pre] =  N[rigth][pre] + 1
            loss_acc = [r / test_batch_count for r in loss_acc]
            print('loss and acc:', loss_acc)
            print(N)
            print( [N[i][i] for i in range(len(N))] )

            save_path = ckpt_dir+'/'+'lib_resnet_%d'
            saver.save(sess, save_path % epoch)
