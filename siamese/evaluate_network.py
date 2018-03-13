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
from test_data_orchestrator import TestDataOrchestrator
import metrics

from tensorflow.contrib import slim
model = 'siamese_finetuned/model'
print("begin tensor session")
batch_size = 1
test_file = '../data/test_with_neg.txt'
test_orch = TestDataOrchestrator(test_file)
test_batches_per_epoch = np.floor(test_orch.dataset_size / batch_size).astype(np.int16)
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
    d = tf.reduce_sum(tf.square(report_features - satelite_features), 1)
    d_sqrt = tf.sqrt(d)

    print("{} testing".format(datetime.now()))
    total_labels = None
    total_logits = None
    for step in range(test_batches_per_epoch):
        batch_t_reports, batch_t_satelites, batch_t_labels = test_orch.get_next_testing_batch(batch_size)
        logits = sess.run(d_sqrt, feed_dict={reports: batch_t_reports, satelites: batch_t_satelites, labels:batch_t_labels, train_phase:False})
        if step == 0:
            total_logits = logits
            total_labels = batch_t_labels
        else:
            total_logits = np.concatenate((total_logits, logits), axis=0)
            total_labels = np.concatenate((total_labels, batch_t_labels), axis=0)
    total_labels = total_labels.astype(int)
    aucPR, average_precision, roc_auc = metrics.get_metrics(total_labels,total_logits)
    print('aucPR: '+str(aucPR))
    print('average_precision: '+str(average_precision))
    print('roc_auc: '+str(roc_auc))
    metrics.plot_roc(total_labels,total_logits)
    metrics.plot_pr(total_labels,total_logits)
    np.savetxt('results/total_labels.out', total_labels, delimiter='\n') 
    np.savetxt('results/total_logits.out', total_logits, delimiter='\n') 






