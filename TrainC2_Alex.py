
import dataset
import tensorflow as tf
import numpy as np
from numpy.random import seed
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'
seed(10)
from tensorflow import set_random_seed

set_random_seed(20)

batch_size = 128
classes = ['r1', 'r2']
num_classes = len(classes)

validation_size = 0.2
img_size = 48
num_channels = 2
train_path = '/media/data2/mhy/data/0918'

session = tf.Session()
data = dataset.read_train_sets(train_path, img_size, classes, validation_size)
print("Complete reading input data.Will Now print a snippet of it")
# print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
# np.save("TempData.npy",data)
# data=np.load("TempData.npy")



# session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
filter_size_conv1 = 11
num_filters_conv1 = 96

filter_size_conv2 = 5
num_filters_conv2 = 256

filter_size_conv3 = 3
num_filters_conv3 = 384

filter_size_conv4 = 3
num_filters_conv4 = 384

filter_size_conv5 = 3
num_filters_conv5 = 256

# 全连接层的输出
fc_layer_size1 = 4096
fc_layer_size2 = 4096


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolution_layer(input,
                             num_input_channels,
                             conv_filter_size,
                             num_filters,
                             use_maxpool=True):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    layer += biases
    layer = tf.nn.relu(layer)
    if use_maxpool:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.dropout(layer, keep_prob=0.8)
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


layer_conv1 = create_convolution_layer(input=x,
                                       num_input_channels=num_channels,
                                       conv_filter_size=filter_size_conv1,
                                       num_filters=num_filters_conv1,)
layer_conv2 = create_convolution_layer(input=layer_conv1,
                                       num_input_channels=num_filters_conv1,
                                       conv_filter_size=filter_size_conv2,
                                       num_filters=num_filters_conv2)
layer_conv3 = create_convolution_layer(input=layer_conv2,
                                       num_input_channels=num_filters_conv2,
                                       conv_filter_size=filter_size_conv3,
                                       num_filters=num_filters_conv3,
                                       use_maxpool=False)
layer_conv4 = create_convolution_layer(input=layer_conv3,
                                       num_input_channels=num_filters_conv3,
                                       conv_filter_size=filter_size_conv4,
                                       num_filters=num_filters_conv4,
                                       use_maxpool=False)
layer_conv5 = create_convolution_layer(input=layer_conv4,
                                       num_input_channels=num_filters_conv4,
                                       conv_filter_size=filter_size_conv5,
                                       num_filters=num_filters_conv5,)
layer_flat = create_flatten_layer(layer_conv5)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size1,
                            use_relu=True)
layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=layer_fc1.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size2,
                            use_relu=True)
layer_fc3 = create_fc_layer(input=layer_fc2,
                            num_inputs=fc_layer_size2,
                            num_outputs=num_classes,
                            use_relu=False)
y_pred = tf.nn.softmax(layer_fc3, name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

global_step=tf.Variable(0,trainable=False)
initial_learning_rate=1e-4
learning_rate=tf.train.exponential_decay(initial_learning_rate,
                                         global_step=global_step,
                                         decay_steps=500,
                                         decay_rate=0.1)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, loss, val_loss, i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    print("epoch:", str(epoch + 1) + ",i:", str(i) +
          ",acc:", str(acc) + ",val_acc:", str(val_acc) + ",loss:", str(loss) + ",val_loss:", str(val_loss))


total_iterations = 0
saver = tf.train.Saver()


def train(num_iteration):
    global total_iterations
    for i in range(total_iterations, total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)
        examples = data.train.num_examples()
        if i % 100 == 0:
            loss = session.run(cost, feed_dict=feed_dict_tr)
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, loss, val_loss, i)
            saver.save(session, '/media/data2/mhy/models/model0/SimpleFusion.ckpt', global_step=i)
    total_iterations += num_iteration


train(num_iteration=50001)