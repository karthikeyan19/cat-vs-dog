import threading
from queue import Queue
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

TRAIN_DIR = 'D:/Python_tutorial/ML/cat_vs_dog/train'
TEST_DIR = 'D:/Python_tutorial/ML/cat_vs_dog/test1'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match
images = os.listdir(TRAIN_DIR)
data = []

def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]

def create_train_data(ranges):
    
    training_data = []
    start = ranges - 2500;
    for i in range(start, ranges):
        img = images[i]
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
##    for img in tqdm(os.listdir(TRAIN_DIR)):
##        label = label_img(img)
##        path = os.path.join(TRAIN_DIR,img)
##        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
##        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
##        training_data.append([np.array(img),np.array(label)])
    
    return training_data

def threader():
    while True:
        # gets an worker from the queue
        worker = q.get()

        # Run the example job with the avail worker in queue (thread)
        data.append(create_train_data((worker+1)*2500))
        # completed with the job
        q.task_done()


q = Queue()
for x in range(10):

    t = threading.Thread(target=threader)

    t.daemon = True

    t.start()

for worker in range(10):
    q.put(worker)


q.join()

trained_data = []
for i in range(10):
    for j in data[i]:
        trained_data.append(j)
print(len(trained_data))    
shuffle(trained_data)
np.save('trained_data1.npy', trained_data)

    


