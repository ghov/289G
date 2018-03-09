import cv2
import numpy as np
import random
import math


POS_SIZE=1
ZOOM_IN_ONE_FRACTION = 0.05
ZOOM_IN_TWO_FRACTION = 0.10
data_directory = '/Users/nicholasjoodi/Documents/ucdavis/computerScience/VisualRecognition/diagrams_with_google_label/'
class TestDataOrchestrator:
    def __init__(self, data_file, should_crop=True, scale_size=(224, 224),\
     report_mean = np.array([245.57, 245.63, 245.59]),\
     satelite_mean = np.array([ 89.18, 104.17, 103.25])):
        self.should_crop=should_crop
        self.scale_size=scale_size
        self.data_index = 0
        self.index_data(data_file)
        self.report_mean = report_mean
        self.satelite_mean = satelite_mean


    def index_data(self,data_file):
        with open(data_file) as _file:
            rows = _file.readlines()
            self.dataset_size = len(rows)
            self.reports = []
            self.satelites = []
            self.labels = []
            for r in rows:
                images = r.strip().split()
                self.reports.append(images[0])
                self.satelites.append(images[1])
                self.labels.append(float(images[2]))

    def crop_y(self,img):
        height = np.size(img,0)
        top_precentage = .16
        bottom_percentage = .1
        y_start = math.floor(height*top_precentage)
        marker = math.floor(height*bottom_percentage)
        y_end = math.floor(height-marker)
        crop_img = img[y_start:y_end, :]
        return crop_img



    def show_image(self,image ):
        cv2.imshow("cropped", image)
        cv2.waitKey(0)



    def reset_data_index(self):
        self.data_index = 0


    def get_next_testing_batch(self,batch_size):
        report_image_paths = self.reports[self.data_index:self.data_index + batch_size]
        satelite_image_paths = self.satelites[self.data_index:self.data_index + batch_size]
        the_labels = self.labels[self.data_index:self.data_index + batch_size]
        self.data_index+=batch_size
        reports = []
        satelites = []
        labels = []

        for i in range(len(report_image_paths)):
            print(report_image_paths[i])
            label = [the_labels[i]]
            report_correct = cv2.imread(data_directory+report_image_paths[i])
            satelites_total = cv2.resize(cv2.imread(data_directory+satelite_image_paths[i]),(self.scale_size[0], self.scale_size[1]))
            if self.should_crop:
                report_correct = self.crop_y(report_correct)
            report_correct = cv2.resize(report_correct,(self.scale_size[0], self.scale_size[1]))
            report_correct =  [(report_correct-self.report_mean)]
            satelites_total =  [(satelites_total-self.satelite_mean)]    
            reports = reports + report_correct 
            satelites = satelites + satelites_total
            labels = labels + label
        reports = np.array(reports)
        satelites = np.array(satelites)
        labels = np.array(labels)
        labels= labels.reshape((np.shape(labels)[0],1))
        return reports, satelites, labels





            