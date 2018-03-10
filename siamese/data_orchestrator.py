import cv2
import numpy as np
import random
import math


POS_SIZE=1
ZOOM_IN_ONE_FRACTION = 0.05
ZOOM_IN_TWO_FRACTION = 0.10
data_directory = '/Users/nicholasjoodi/Documents/ucdavis/computerScience/VisualRecognition/diagrams_with_google_label/'
class DataOrchestrator:
    def __init__(self, data_file, should_crop=True, corruption_size=20, shuffle=True, scale_size=(224, 224),\
     report_mean = np.array([245.57, 245.63, 245.59]),\
     satelite_mean = np.array([ 89.18, 104.17, 103.25])):
        self.should_crop=should_crop
        self.shuffle=shuffle
        self.scale_size=scale_size
        self.data_index = 0
        self.index_data(data_file)
        self.report_mean = report_mean
        self.satelite_mean = satelite_mean

        self.corruption_size = corruption_size
        if self.shuffle:
            self.shuffle_data()


    def index_data(self,data_file):
        with open(data_file) as _file:
            rows = _file.readlines()
            self.dataset_size = len(rows)
            self.reports = []
            self.satelites = []
            for r in rows:
                images = r.strip().split()
                self.reports.append(images[0])
                self.satelites.append(images[1])

    def shuffle_data(self):
        reports = self.reports.copy()
        satelites = self.satelites.copy()
        self.reports = []
        self.satelites = []
        indices = np.random.permutation(self.dataset_size)
        for i in indices:
            self.reports.append(reports[i])
            self.satelites.append(satelites[i])

    def crop_y(self,img):
        height = np.size(img,0)
        top_precentage = .16
        bottom_percentage = .1
        y_start = math.floor(height*top_precentage)
        marker = math.floor(height*bottom_percentage)
        y_end = math.floor(height-marker)
        crop_img = img[y_start:y_end, :]
        return crop_img

    def crop(self,img, percentage):
        height = np.size(img,0)
        width = np.size(img,1)
        y_start = math.floor(height*percentage)
        y_end = math.floor(height-y_start)
        x_start= math.floor(width*percentage)
        x_end= math.floor(width-x_start)
        crop_img = img[y_start:y_end, x_start:x_end]
        return crop_img

    def rotate_image(self,image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def show_image(self,image ):
        cv2.imshow("cropped", image)
        cv2.waitKey(0)

    def do_image_modification(self,image):
        # self.show_image(image)
        rando = np.random.random()
        if rando <= 0.125:
            image = self.rotate_image(image,45.0)
        elif rando > 0.125 and rando<=0.25:
            image = self.rotate_image(image,90.0)
        elif rando > 0.25 and rando<=0.375:
            image = self.rotate_image(image,135.0)
        elif rando > 0.375 and rando<=0.5:
            image = self.rotate_image(image,180.0)
        elif rando > 0.5 and rando<=0.625:
            image = self.rotate_image(image,225.0)
        elif rando > 0.625 and rando<=0.75:
            image = self.rotate_image(image,270.0)
        elif rando > 0.75 and rando<=0.875:
            image = self.rotate_image(image,315.0)
        # self.show_image(image)
        rando = np.random.random()
        if rando <= 0.3:
            image = self.crop(image,ZOOM_IN_ONE_FRACTION)
        elif rando <= 0.6 and rando > 0.3:
            image = self.crop(image,ZOOM_IN_TWO_FRACTION)
        # self.show_image(image)
        return image


    def reset_data_index(self):
        self.data_index = 0
        if self.shuffle:
            self.shuffle_data()


    def get_next_training_batch(self,batch_size):
        report_image_paths = self.reports[self.data_index:self.data_index + batch_size]
        satelite_image_paths = self.satelites[self.data_index:self.data_index + batch_size]
        self.data_index+=batch_size
        batch_size*=self.corruption_size
        batch_size+=POS_SIZE
        reports = []
        satelites = []
        labels = []

        for i in range(len(report_image_paths)):
            # print(report_image_paths[i])
            # print(satelite_image_paths[i])
            label = [ 0.0 if j < POS_SIZE else 1.0 for j in range(POS_SIZE+ self.corruption_size)]
            satelites_total = [cv2.resize(cv2.imread(data_directory+satelite_image_paths[i]),(self.scale_size[0], self.scale_size[1])) for j in range(POS_SIZE)]
            report_correct = [cv2.imread(data_directory+report_image_paths[i]) for j in range((self.corruption_size +POS_SIZE))]
            satelites_total_corrupt = [cv2.resize(cv2.imread(data_directory+self.satelites[j]),(self.scale_size[0], self.scale_size[1])) for j in random.sample(range( len(self.satelites)), self.corruption_size)]
            if self.should_crop:
                report_correct = [self.crop_y(r) for r in report_correct]
            for i in range(len(report_correct)):
                report_correct[i] = self.do_image_modification(report_correct[i])
            report_correct = [cv2.resize(report_correct[j],(self.scale_size[0], self.scale_size[1])) for j in range((self.corruption_size +POS_SIZE))]
            report_correct = [ (report_correct[j]-self.report_mean) for j in range((self.corruption_size +POS_SIZE))]
            satelites_total = [ (satelites_total[j]-self.satelite_mean) for j in range(POS_SIZE)]    
            satelites_total_corrupt = [ (satelites_total_corrupt[j]-self.satelite_mean) for j in range(self.corruption_size)]    
            reports = reports + report_correct
            satelites = satelites + satelites_total + satelites_total_corrupt
            labels = labels + label
        reports = np.array(reports)
        satelites = np.array(satelites)
        labels = np.array(labels)
        labels= labels.reshape((np.shape(labels)[0],1))
        # print(np.shape(reports))
        # print(np.shape(satelites))
        # print(np.shape(labels))
        return reports, satelites, labels





            