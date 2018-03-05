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
import data

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
    reports = tf.placeholder(tf.float32, [None, image_size,image_size,3], name='reports')
    satelites = tf.placeholder(tf.float32, [None, image_size,image_size,3],name='satelites')
    labels = tf.placeholder(tf.float32, [None, 1],name='labels')
    # dataset = data.get_data('../dataset/train/')
    # satelites,reports, labels = load_batch(dataset,batch_size=1, height=image_size, width=image_size)
    # reports, satelites, labels = get_sample_train_data()
    output_features = 4096
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        # logits, _ = inception.inception_v1(reports, num_classes=output_features, is_training=True, scope='InceptionV1')
        report_features, _ = inception.inception_v1(reports, num_classes=output_features, is_training=True, scope='InceptionV1')

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        # logits_2, _ = inception.inception_v1(satelites, num_classes=output_features, is_training=True, scope='InceptionV2')
        satelite_features, _ = inception.inception_v1(satelites, num_classes=output_features, is_training=True, scope='InceptionV2')


    margin = 0.2


    d = tf.reduce_sum(tf.square(report_features - satelite_features), 1)
    d_sqrt = tf.sqrt(d)
    loss = labels * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - labels) * d

    loss = 0.5 * tf.reduce_mean(loss)


    # Create some summaries to visualize the training process:
    tf.summary.scalar('losses/TotalLoss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(loss, name='optimizer')
    # Specify the optimizer and create the train op:
    init_all = tf.global_variables_initializer()
    saver = tf.train.Saver()
    init = get_init_fn()
    sess = tf.Session()

    sess.run(init_all)
    print('before incorporating pretrained weights - example weight:InceptionV1/Mixed_5c/Branch_3')
    for var in slim.get_model_variables():
        if var.op.name.startswith("InceptionV1/Mixed_5c/Branch_3"):
            print(sess.run(var))
            break
    init(sess)
    print('after incorporating pretrained weights - example weight:InceptionV1/Mixed_5c/Branch_3')
    for var in slim.get_model_variables():
        if var.op.name.startswith("InceptionV1/Mixed_5c/Branch_3"):
            print(sess.run(var))
            break
    tf.add_to_collection('report_features', report_features)
    tf.add_to_collection('satelite_features', satelite_features)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('optimizer', optimizer)
    saver.save(sess,train_dir+'model')

