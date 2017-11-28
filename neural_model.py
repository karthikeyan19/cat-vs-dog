import tflearn
import numpy as np
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os                  # dealing with directories
import tensorflow as tf
import matplotlib.pyplot as plt

train_data = np.load('train_data1.npy')

IMG_SIZE = 64
LR = 1e-3
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '1_conv-basic-video') # just so we remember which saved model is which, sizes must match

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')


convnet = conv_2d(convnet, 32, 3, activation='relu')
#convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
#convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
#convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 256, 3, activation='relu')
#convnet = max_pool_2d(convnet, 3)
convnet = tflearn.global_avg_pool(convnet)
convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-1000]
test = train_data[-1000:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)




# if you need to create the data:
#test_data = process_test_data()
# if you already have some saved:
test_data = np.load('test_data1.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

from tqdm import tqdm
with open('submission3_file.csv','w') as f:
    f.write('id,label\n')
    
            
with open('submission3_file.csv','a') as f:
    for data in test_data:
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))


