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
epochs = 1
batch_size = 2
display_step = 1
train_file = '../data/train.txt'
train_orch = DataOrchestrator(train_file, shuffle = True)
batches_per_epoch = np.floor(train_orch.dataset_size / batch_size).astype(np.int16)
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
    for epoch in range(epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        step = 1
        while step < batches_per_epoch:
            batch_reports, batch_satelites, batch_labels = train_orch.get_next_training_batch(batch_size)
            batch_labels= batch_labels.reshape((np.shape(batch_labels)[0],1))
            _, current_cost= sess.run([optimizer, loss], feed_dict={reports: batch_reports, satelites: batch_satelites, labels:batch_labels})
            print(current_cost)





