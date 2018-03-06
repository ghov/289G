import cv2
import numpy as np
import math

def crop_y(img):
    height = np.size(img,0)
    top_precentage = .16
    bottom_percentage = .1
    y_start = math.floor(height*top_precentage)
    marker = math.floor(height*bottom_percentage)
    y_end = math.floor(height-marker)
    crop_img = img[y_start:y_end, :]
    return crop_img

def crop(img, percentage):
    height = np.size(img,0)
    width = np.size(img,1)
    y_start = math.floor(height*percentage)
    y_end = math.floor(height-y_start)
    x_start= math.floor(width*percentage)
    x_end= math.floor(width-x_start)
    crop_img = img[y_start:y_end, x_start:x_end]
    return crop_img

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

train_file = 'train.txt'
test_file = '/Users/nicholasjoodi/Documents/ucdavis/computerScience/VisualRecognition/diagrams_with_google_label/9345-2015-0615/31697-9345-2015-0615_3-3_3.jpeg'
default_image_size = 224
stats_file = 'stats.txt'
img_report = cv2.imread(test_file)
crop_img = crop_y(img_report)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
crop_img = crop(crop_img, 0.05)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

crop_img = crop(crop_img, 0.1)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

crop_img = rotateImage(crop_img, 90.0)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

crop_img = rotateImage(crop_img, -45.0)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

crop_img = rotateImage(crop_img, -45.0)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

crop_img = rotateImage(crop_img, -45.0)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

crop_img = rotateImage(crop_img, -45.0)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
