from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.image as mpimg
from datetime import datetime

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
from data_orchestrator import DataOrchestrator

from tensorflow.contrib import slim
image_size = inception.inception_v1.default_image_size

model = 'siamese_finetuned/model'
print("begin tensor session")
batches = 50
batch_size = 40
size = batches *batch_size
save_step = 5
train_file = '../data/train.txt'
train_orch = DataOrchestrator(train_file, shuffle = True)
batches_per_epoch = np.floor(train_orch.dataset_size / batch_size).astype(np.int16)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
report_mean = np.array([245.57, 245.63, 245.59])
satelite_mean = np.array([ 89.18, 104.17, 103.25])
with tf.Session(config = config) as sess:
    saver = tf.train.import_meta_graph(model+'.meta')
    saver.restore(sess, model)
    graph = tf.get_default_graph()
    reports = graph.get_tensor_by_name("reports:0")
    train_phase = graph.get_tensor_by_name("train_phase:0")
    satelites = graph.get_tensor_by_name("satelites:0")
    optimizer =  tf.get_collection('optimizer')[0]
    loss =  tf.get_collection('loss')[0]
    report_features = tf.get_collection('report_features')[0]
    satelite_features = tf.get_collection('satelite_features')[0]

    print("Begin training...")
    step = 0
    while step < batches:
        batch_reports, batch_satelites = train_orch.get_next_training_batch_without_corruption(batch_size)
        r_features, s_features= sess.run([report_features, satelite_features], feed_dict={reports: batch_reports, satelites: batch_satelites,  train_phase:False})
        if step == 0:
            np_reports = r_features
            np_satelites = s_features
        else:
            np_reports = np.concatenate((np_reports, r_features), axis=0)
            np_satelites = np.concatenate((np_satelites, s_features), axis=0)
        step += 1

    distance = np.empty((size))
    for i in range(size):
        img_report = np_reports[i]
        img_satelite = np_satelites[i]
        distance[i] = np.linalg.norm(img_report-img_satelite)
    mean_dist = np.mean(distance)
    print(mean_dist)


#550