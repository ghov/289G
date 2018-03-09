import cv2
import numpy as np
import random
import math

POS_SIZE=1
out_file ='test_with_neg.txt'
class CreateTestNeg:
    def __init__(self, data_file, corruption_size=20, shuffle=True):
        self.data_index = 0
        self.index_data(data_file)

        self.corruption_size = corruption_size


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

    def run(self):
        reports = []
        satelites = []
        labels = []
        for i in range(len(self.reports)):
            print(self.reports[i])
            label = [ 0.0 if j < POS_SIZE else 1.0 for j in range(POS_SIZE+ self.corruption_size)]
            satelites_total= [self.satelites[j] for j in range(POS_SIZE)]
            satelites_total_corrupt = [self.satelites[j] for j in random.sample(range( len(self.satelites)), self.corruption_size)]
            report_correct  = [self.reports[i] for j in range((self.corruption_size +POS_SIZE))]

            reports = reports + report_correct
            satelites = satelites + satelites_total + satelites_total_corrupt
            labels = labels + label

        with open(out_file, 'w') as _file:
            for i in range(len(reports)):
                _file.write("%s\t%s\t%s\n" % (reports[i],satelites[i],labels[i]))


if __name__ == "__main__":

    orch = CreateTestNeg('test.txt', shuffle = False)
    orch.run()


            