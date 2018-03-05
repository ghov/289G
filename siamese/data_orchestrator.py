import cv2
import numpy as np
import random

POS_SIZE=5
ZOOM_IN_ONE_FRACTION = 0.05
ZOOM_IN_TWO_FRACTION = 0.10
class DataOrchestrator:
    def __init__(self, data_file, crop_y=False, corruption_size=15, shuffle=False,\
     scale_size=(224, 224), report_mean = np.array([245.8, 245.8, 245.8]), report_std = np.array([39.6, 39.5, 39.5])
     \ satelite_mean = np.array([93.0, 110.0, 109.8]), satelite_std = np.array([44.1, 47.5, 48.5]) ):
        self.crop_y=crop_y
        self.zoom=zoom
        self.shuffle=shuffle
        self.scale_size=scale_size
        self.data_index = 0
        self.index_data(data_file)
        self.report_mean = report_mean
        self.report_std = report_std
        self.satelite_mean = satelite_mean
        self.satelite_std = satelite_std

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
        satelites = self.labels.copy()
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
        reports = [] # np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        satelites = [] #np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        labels = []

        for i in range(len(report_image_paths)):
            label = [ 0.0 if j < POS_SIZE else 1.0 for j in range(POS_SIZE+ self.corruption_size)]
            report_correct = [cv2.resize(cv2.imread(report_image_paths[i]),(self.scale_size[0], self.scale_size[1])) for j in range(POS_SIZE)]
            report_corrupt = [cv2.resize(cv2.imread(report_image_paths[j]),(self.scale_size[0], self.scale_size[1])) for j in random.sample(range(0, len(report_image_paths)), self.corruption_size)]
            satelites_total = [cv2.resize(cv2.imread(satelite_image_paths[i]),(self.scale_size[0], self.scale_size[1])) for j in range((self.corruption_size +POS_SIZE))]
            if self.crop_y:
                report_correct = [self.crop_y(r) for r in report_correct]
                report_corrupt = [self.crop_y(r) for r in report_corrupt]
            for i in range(len(report_correct)):
                if i == 1:
                    report_correct[i] = self.rotate_image(report_correct[i],-45)
                elif i  == 2:
                    report_correct[i] = self.rotate_image(report_correct[i],-90)
                elif i  == 3:
                    report_correct[i] = self.rotate_image(report_correct[i],45)
                elif i  == 4:
                    report_correct[i] = self.rotate_image(report_correct[i],90)
                if np.random.random() <= 0.3:
                    report_correct[i] = self.crop(report_correct[i],ZOOM_IN_ONE_FRACTION)
                elif np.random.random() <= 0.6 and np.random.random() > 0.3:
                    report_correct[i] = self.crop(report_correct[i],ZOOM_IN_TWO_FRACTION)
            reports = reports + report_correct + report_corrupt
            satelites = satelites + satelites_total
            labels = labels + label
        return np.array(reports), np.array(satelites), np.array(labels)


            