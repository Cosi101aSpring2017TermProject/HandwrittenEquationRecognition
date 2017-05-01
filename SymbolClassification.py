import LocalHandwrittenSymbolDataset
import SymbolSegmentor
from os import listdir
import skimage
import skimage.util
import skimage.transform
from skimage import exposure
import tensorflow as tf
import argparse
import sys
import numpy
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

do_restore = 0

class ClassificationDictionary:

    classification_dict = dict()

    def __init__(self, training_dir: str):
        # './annotated'
        if training_dir == "":
            return
        files = [f for f in listdir(training_dir)]
        classification = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])

        for f in files:
            a = f.split("_")
            if len(a) == 8:
                print(f)
                print(a[3])
                classification.add(a[3])
        sorted_classification = numpy.array(sorted(classification))
        print("xxx")
        print(sorted_classification)
        print("there are %d kinds of symbols" % len(classification))

        for i in range(0, len(classification)):
            self.classification_dict[sorted_classification[i]] = i

        print(self.classification_dict)

    def get_classification_array(self, symbol):
        if len(self.classification_dict) == 0:
            print("You cannot call get_classification_array(symbol) for now, \n the classification_dict is empty.")
            return numpy.array([])
        zeros_array = [0.] * len(self.classification_dict)
        index = self.classification_dict[symbol]
        if index is None:
            print("cannot identify given symbol " + symbol)
            print("return all %d zeros" % len(self.classification_dict))
            return zeros_array
        else:
            zeros_array[index] = 1.0
            return zeros_array

    def get_classes_number(self):
        return len(self.classification_dict)

    def convert_mnist(self, mnist_class_array):
        i = 0
        for num in mnist_class_array:
            if num == 1.:
                return self.get_classification_array(str(i))
            i += 1
        return self.get_classification_array("nothing")


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# This method stretch the image
def stretch(raw_image, scale, compression_flag):
    stretched_array = numpy.reshape(raw_image, (28, 28))
    # plot show image
    # plt.imshow(stretched_array)
    # plt.show()
    if compression_flag:
        stretched_array = numpy.repeat(numpy.repeat(stretched_array, 30, axis=0), 30+scale*3, axis=1)
    else:
        stretched_array = numpy.repeat(numpy.repeat(stretched_array, 30+scale*3, axis=0), 30, axis=1)
    (vertical_pixel, horizontal_pixel) = stretched_array.shape
    if vertical_pixel > horizontal_pixel:
        vertical_padding = 0
        horizontal_padding = int(round((vertical_pixel - horizontal_pixel) / 2))
        padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
        stretched_array = numpy.lib.pad(stretched_array, padding, 'constant', constant_values=0)

    else:
        horizontal_padding = 0
        vertical_padding = int(round((horizontal_pixel - vertical_pixel) / 2))
        padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
        stretched_array = numpy.lib.pad(stretched_array, padding, 'constant', constant_values=0)
    stretched_im = numpy.resize(stretched_array, (28, 28))
    stretched_im_784 = numpy.reshape(stretched_im, (1, 784))
    # plot show image
    # plt.imshow(stretched_im)
    # plt.show()
    return stretched_im_784[0].tolist()


def rot(raw_image, scale):
    stretched_array = numpy.reshape(raw_image, (28, 28))
    # plot show image
    # plt.imshow(stretched_array)
    # plt.show()
    stretched_array = numpy.repeat(numpy.repeat(stretched_array, 30, axis=0), 30, axis=1)
    count = -40 * scale
    for k in range(0, 40):
        for l in range(0, 21):
            for m in range(0, 840):
                if m+count >= 0 and m+count < 840:
                    if count < 0:
                        tmp = stretched_array[k*21+l][840-m+count]
                        stretched_array[k*21+l][840-m] = tmp
                    else:
                        tmp = stretched_array[k*21+l][m+count]
                        stretched_array[k*21+l][m] = tmp
        count = count + 2 * scale
        # print(count)
    rotated_im = numpy.resize(stretched_array, (28, 28))
    rotated_im_784 = numpy.reshape(rotated_im, (1, 784))
    # plot show image
    # plt.imshow(stretched_im)
    # plt.show()
    # print("in rot")
    # print(rotated_im_784.shape)
    return rotated_im_784[0].tolist()


# def convert_dimension(py_list):
#     temp = numpy.array([])
#     for l in py_list:
#         print(l[0])
#         print(len(l[0]))
#         print(temp.size)
#         # print(numpy.array(l[0]).shape)
#         if temp.size == 0:
#             temp = numpy.array(l[0])
#         else:
#             print("add")
#             numpy.vstack((temp, numpy.array(l[0])))
#     print(temp.shape)
#     print(temp)
#     return temp


def main(_):

    classficationDic = ClassificationDictionary('./annotated')
    localData = LocalHandwrittenSymbolDataset.LocalSymbolData(classficationDic)
    testing_data = SymbolSegmentor.SymbolSegmentor()

    # DEFINE MODEL
    number_of_classes = classficationDic.get_classes_number()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, number_of_classes])
    W = tf.Variable(tf.zeros([784, number_of_classes]))
    b = tf.Variable(tf.zeros([number_of_classes]))
    y = tf.matmul(x, W) + b
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, number_of_classes])
    b_fc2 = bias_variable([number_of_classes])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    results = tf.argmax(y_conv, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # save training
    #TODO: TRAIN
    if do_restore == 1:
        saver.restore(sess, './model')
    else:
        # Train
        number_steps = 150
        for i in range(number_steps):  # 20000
            # get the data
            batch = mnist.train.next_batch(50)
            localdata_batch = localData.next_batch(50)
            label_batch = [classficationDic.convert_mnist(num_array) for num_array in batch[1]]
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: label_batch, keep_prob: 1.0})

            train_step.run(feed_dict={x: batch[0], y_: label_batch, keep_prob: 0.5})

            # train with mnist
            if localdata_batch[0].size != 0:
                train_step.run(feed_dict={x: localdata_batch[0], y_: localdata_batch[1], keep_prob: 0.5})
                # train with different param
                for rot_param in range(-2, 3):
                    train_step.run(
                        feed_dict={x: [rot(num_array, rot_param) for num_array in localdata_batch[0]],
                                   y_: localdata_batch[1],
                                   keep_prob: 0.5})
                for stretch_param in range(1, 4):
                    train_step.run(
                        feed_dict={x: [stretch(num_array, stretch_param, True) for num_array in batch[0]],
                                   y_: localdata_batch[1], keep_prob: 0.5})
                    train_step.run(
                        feed_dict={x: [stretch(num_array, stretch_param, False) for num_array in batch[0]],
                                   y_: localdata_batch[1], keep_prob: 0.5})
            train_loc_acc = accuracy.eval(feed_dict={x: localdata_batch[0], y_: localdata_batch[1], keep_prob: 1.0})
            print("-`-`-`-`-`-`-`-`-`")
            print("step %d out of %d, \nMNIST data training accuracy %g" % (i, number_steps, train_accuracy))
            print("Local data training accuracy %g" % train_loc_acc)
            print("-`-`-`-`-`-`-`-`-`\n")

        save_path = saver.save(sess, 'model')
        print("Model saved in file: %s" % save_path)


    #TODO: TEST




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)