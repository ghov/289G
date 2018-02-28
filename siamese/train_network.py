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
import cv2

from tensorflow.contrib import slim
image_size = inception.inception_v1.default_image_size
def get_sample_train_data():


    img = cv2.imread('../sample_data/report.jpeg')
    np_img = np.array(img)


    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32)
    copy = np.empty_like (img)
    copy[:] = img
    np_satelite_pos = cv2.resize(cv2.imread('../sample_data/pos_google.png'),(image_size, image_size))
    np_satelite_pos = np_satelite_pos.astype(np.float32)
    np_satelite_neg = cv2.resize(cv2.imread('../sample_data/neg_google.png'),(image_size, image_size))
    np_satelite_neg = np_satelite_neg.astype(np.float32)
    reports = np.stack((img,copy))
    satelites = np.stack((np_satelite_pos,np_satelite_neg))

    labels = np.ones((2,1))
    labels[0,0] = 0.0
    # labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    return reports, satelites, labels

model = 'siamese_finetuned/model'
print("begin tensor seesion")
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model+'.meta')
    saver.restore(sess, model)
    graph = tf.get_default_graph()
    reports = graph.get_tensor_by_name("reports:0")
    satelites = graph.get_tensor_by_name("satelites:0")
    labels = graph.get_tensor_by_name("labels:0")
    optimizer =  tf.get_collection('optimizer')[0]
    loss =  tf.get_collection('loss')[0]
    report_features = tf.get_collection('report_features')[0]
    satelite_features = tf.get_collection('satelite_features')[0]


    print("Begin training...")
    _reports, _satelites, _labels = get_sample_train_data()
    _, current_cost= sess.run([optimizer, loss], feed_dict={reports: _reports, satelites: _satelites, labels:_labels})
    print(current_cost)





