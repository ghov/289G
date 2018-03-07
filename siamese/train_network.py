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
epochs = 50
batch_size = 10
save_step = 5
train_file = '../data/train.txt'
train_orch = DataOrchestrator(train_file, shuffle = True)
batches_per_epoch = np.floor(train_orch.dataset_size / batch_size).astype(np.int16)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    saver = tf.train.import_meta_graph(model+'.meta')
    saver.restore(sess, model)
    graph = tf.get_default_graph()
    reports = graph.get_tensor_by_name("reports:0")
    train_phase = graph.get_tensor_by_name("train_phase:0")
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
            _, current_cost= sess.run([optimizer, loss], feed_dict={reports: batch_reports, satelites: batch_satelites, labels:batch_labels, train_phase:True})
            print(current_cost)
            if step%save_step == 0:
                saver.save(sess,model)
                print('Saved model parameters')
            step += 1





