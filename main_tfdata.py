from models.resnet import ResNet50
from datasets.augLib import  class_map, class_num, _parse_example

import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

training_filenames = [r'F:\data\augLib\auglib_rain.tfrecord']
test_filenames = [r'F:\data\augLib\auglib_test.tfrecord']

learning_rate = 0.01
batch_size = 24
ckpt_dir = 'ckpt/lib_resnet50'



def _parse_function(example_proto):
    return _parse_example(example_proto,shape=[224,224,1])


def validate(filename_list, session, ops):
    if not isinstance(filename_list,list):
        filename_list = [filename_list]
    print('strat to validate ',filename_list)
    session.run(iterator.initializer, feed_dict={filenames: filename_list})
    step = 0
    loss_acc = [0,0]

    while True:
        try:
            logits_val, loss_val, acc_val = session.run(ops)
            if step % 50 == 0:
                print('step %d >>> loss:%f, acc:%f' % (step, loss_val, acc_val))

            loss_acc = [ r+t for (r, t) in zip(loss_acc,[loss_val,acc_val])]
            step += 1
        except tf.errors.OutOfRangeError:
            loss_acc = [ r/step for r in loss_acc]
            print('loss and acc:',loss_acc)
            break
    print('=========val end==========')


# 如果您有两组分别用于训练和验证的文件，
# 则可以使用 tf.placeholder(tf.string) 来表示文件名，
# 并使用适当的文件名初始化迭代器：
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)  # Parse the record into tensors.
dataset = dataset.shuffle(buffer_size=10000)
# dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()

next_img, next_label = iterator.get_next()
print(next_img.get_shape())
print(next_label.get_shape())

model = ResNet50(next_img, class_num)
logits = model.logits
print(logits)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=next_label,
                                                                   logits=logits,
                                                                   name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='train_loss')
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

with tf.name_scope('train'):
    train_op = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,momentum=0.9).minimize(loss=loss + l2 * 1e-5)


with tf.name_scope('accuracy'):
    # correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(next_label, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    correct = tf.equal(tf.argmax(logits, 1), tf.cast(next_label,tf.int64))
    # correct = tf.nn.in_top_k(output, next_label, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 又尼玛忘了
    saver = tf.train.Saver(max_to_keep=3)
    # ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    # if ckpt_path and tf.train.checkpoint_exists(ckpt_path):
    #     saver.restore(sess, ckpt_path)
    #     print('restore model from ', ckpt_path)

    # Compute for n epochs.
    print('开始训练')
    # for epoch in range(3):
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    print('初始化迭代器成功')

    img,label = sess.run([next_img,next_label])
    # print(img)
    # print(label)

        # step = 0
        # while True:
        #     try:
        #         _, logits_train, loss_train,  acc_train = sess.run([train_op,logits,loss,accuracy])
        #         if step % 50 == 0:
        #             print('epoch %d, step %d >>> loss:%f, acc:%f' % (epoch,step,loss_train,acc_train))
        #         step += 1
        #     except tf.errors.OutOfRangeError:
        #         break
        # save_path = ckpt_dir + '/' + 'lib-11-100_resnet_%d'
        # saver.save(sess, save_path % epoch)
        #
        # ops = [logits,loss,train_op,accuracy]
        # validate(test_filenames,sess,ops=ops)




