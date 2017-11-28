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



# Building Residual Network

net = tflearn.input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3])

net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)

# Residual blocks

net = tflearn.residual_bottleneck(net, 3, 32, 128)

net = tflearn.residual_bottleneck(net, 2, 64, 256, downsample=True)

net = tflearn.residual_bottleneck(net, 1, 64, 256)

net = tflearn.residual_bottleneck(net, 3, 128, 512, downsample=True)
##
net = tflearn.residual_bottleneck(net, 3, 128, 512)

net = tflearn.batch_normalization(net)

net = tflearn.activation(net, 'relu')

net = tflearn.global_avg_pool(net)

# Regression

net = tflearn.fully_connected(net, 2, activation='softmax')

net = tflearn.regression(net, optimizer='momentum',

                         loss='categorical_crossentropy',

                         learning_rate=0.1)

# Training

model = tflearn.DNN(net, checkpoint_path='model_resnet',

                    max_checkpoints=10, tensorboard_verbose=0)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model.fit(X, Y, n_epoch=30, validation_set=(test_x, test_y),

         snapshot_step=500, batch_size=5, show_metric=True, run_id=MODEL_NAME)






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

