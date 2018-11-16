
from resnet import ResNet50
from densnet_neu import DenseNet
from densenet_bc import DenseNet_BC
# from Data_Augmentation_NEU import get_file
# from Data_Augmentation_NEU import get_nextBatch

from Data_Augmentation_Gaoxian import get_file
from Data_Augmentation_Gaoxian import get_nextBatch
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    IMG_HEIGHT = 144
    IMG_WIDTH = 144
    IMG_DEPTH = 1

    num_class = 6
    batch_size = 12
    learning_rate = 0.0001

    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        ys = tf.placeholder("float", shape=[None, num_class])

        # model = ResNet50(xs,num_class)
        model = DenseNet_BC(xs,num_class,k=32,layers=[6,12,24,16])
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
    TRAIN_DATA_PATH = 'F:/data/img/BaoImage2/train/'
    TEST_DATA_PATH = 'F:/data/img/BaoImage2/test/'
    ckpt_dir = 'gx2_densenet'
    train_data, train_labels = get_file(TRAIN_DATA_PATH)
    test_data, test_labels = get_file(TEST_DATA_PATH)

    batch_count = len(train_data) // batch_size
    print('batch_num:', batch_count)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)
        ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_path and tf.train.checkpoint_exists(ckpt_path):
            saver.restore(sess, ckpt_path)
            print('restore model from ',ckpt_path)

        for epoch in range(1, 1 + 20):
            if epoch == 10:
                learning_rate = learning_rate/10
            if epoch == 15:
                learning_rate = learning_rate/10
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

            save_path = ckpt_dir+'/'+ckpt_dir+'_%d'
            saver.save(sess, save_path % epoch)
