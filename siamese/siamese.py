from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '../')
# sys.path.insert(0, '../pretrained_models')
from pretrained_models import inception_preprocessing
from pretrained_models import inception
import os

from tensorflow.contrib import slim
image_size = inception.inception_v1.default_image_size
batch_size = 3
train_dir = 'siamese_finetuned/'
satelite_inception_network_dir = '../pretrained_models/satelite_inception_network/'
report_inception_network_dir = '../pretrained_models/report_inception_network/'

# The following link provides a good introduction to slim:
#
# https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb
#
# Many ideas were borrowed from that walkthrough for this implementation

def get_sample_train_data():
    img = mpimg.imread('../sample_data/report.jpeg')
    np_img = np.array(img)

    copy = np.empty_like (img)
    copy[:] = img
    tf_img = tf.convert_to_tensor(np_img)
    report=inception_preprocessing.preprocess_image(tf_img,height=image_size, width=image_size)

    report2 = inception_preprocessing.preprocess_image(tf.convert_to_tensor(copy),height=image_size, width=image_size)
    satelite_pos=inception_preprocessing.preprocess_image(tf.convert_to_tensor(np.array(mpimg.imread('../sample_data/pos_google.png'))),height=image_size, width=image_size)
    satelite_neg=inception_preprocessing.preprocess_image(tf.convert_to_tensor(np.array(mpimg.imread('../sample_data/neg_google.png'))),height=image_size, width=image_size)
    reports = tf.stack((report,report2))
    satelites = tf.stack((satelite_pos,satelite_neg))
    labels = np.ones((2,1))
    labels[0,0] = 0
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    return reports, satelites, labels





def get_init_fn():
    checkpoint_exclude_scopes=["InceptionV1/Logits", "InceptionV1/AuxLogits", "InceptionV2"]
    
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    print(exclusions)
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    checkpoint_exclude_scopes=["InceptionV2/Logits", "InceptionV2/AuxLogits", "InceptionV1"]
    
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    print(exclusions)
    variables_to_restore_2 = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore_2.append(var)


    report_init_assign_op, report_init_feed_dict = slim.assign_from_checkpoint(
                os.path.join(report_inception_network_dir, 'inception_v1.ckpt'), variables_to_restore_2, ignore_missing_vars=True)

    satelite_init_assign_op, satelite_init_feed_dict = slim.assign_from_checkpoint(
                os.path.join(satelite_inception_network_dir, 'inception_v1.ckpt'), variables_to_restore, ignore_missing_vars=True)

    def init_fn(sess):
        sess.run(report_init_assign_op, report_init_feed_dict)
        sess.run(satelite_init_assign_op, satelite_init_feed_dict)

    return init_fn




with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)


    reports, satelites, labels = get_sample_train_data()
    output_features = 4096
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(reports, num_classes=output_features, is_training=True, scope='InceptionV1')

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_2, _ = inception.inception_v1(satelites, num_classes=output_features, is_training=True, scope='InceptionV2')


    margin = 0.2

    left_output = logits 
    right_output = logits_2 

    d = tf.reduce_sum(tf.square(left_output - right_output), 1)
    d_sqrt = tf.sqrt(d)
    loss = labels * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - labels) * d

    loss = 0.5 * tf.reduce_mean(loss)

    # Create some summaries to visualize the training process:
    tf.summary.scalar('losses/Total Loss', loss)
  
    # Specify the optimizer and create the train op:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(loss, optimizer)
    
    # Run the training:
    final_loss = slim.learning.train(
        train_op,
        logdir=train_dir,
        init_fn=get_init_fn(),
        number_of_steps=2)       
  
print('Finished training. Last batch loss %f' % final_loss)