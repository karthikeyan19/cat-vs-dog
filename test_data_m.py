import threading
from queue import Queue
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

TRAIN_DIR = 'D:/Python tutorial/ML/cat_vs_dog/train'
TEST_DIR = 'D:/Python_tutorial/ML/cat_vs_dog/test1'
IMG_SIZE = 64
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match
images = os.listdir(TEST_DIR)
data = []

def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]

def create_test_data():
    
      testing_data = []
##    start = ranges - 1250;
##    for i in range(start, ranges):
##        img = images[i]
##        path = os.path.join(TEST_DIR,img)
##        img_num = img.split('.')[0]
##        img = cv2.imread(path,cv2.IMREAD_RGB)
##        img = cv2.imread(path)
##        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
##        testing_data.append([np.array(img), img_num])
##       
      for img in os.listdir(TEST_DIR):
        #label = label_img(img)
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
    
      return testing_data


test_data = create_test_data()
print(len(test_data))    
shuffle(test_data)
np.save('test_data1.npy', test_data)

    


