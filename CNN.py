import numpy as np
from keras import backend as K
from time import time
from keras.layers import *
from keras.models import *
from keras_flops import get_flops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import TensorBoard

# CPU selection
os.environ['CUDA_DEVICE_ORDER']= '[PCI_BUS_ID]'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

# Load Data
datasets, labels = np.load('datasets.npy'), np.load('labels.npy')

# Encode data set
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()

# Test train split
X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2,random_state=0)

# reshape into 1d array
X_train=X_train.reshape(X_train.shape[0],-1,1)
X_test=X_test.reshape(X_test.shape[0],-1,1)
print(X_test.shape, X_train.shape,y_test.shape, y_train.shape)

#SE block
def SE_Block (input_tensor, ratio=16):
    input_shape = K.int_shape(input_tensor)
    squeeze = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    excitation = tf.keras.layers.Dense(units = input_shape[-1]//ratio, kernel_initializer = 'he_normal', activation = 'relu')(squeeze)
    excitation = tf.keras.layers.Dense(units = input_shape[-1], activation = 'sigmoid')(excitation)
    #https://keras.io/api/layers/core_layers/dense/
    scale = tf.keras.layers.Multiply()([input_tensor,excitation])
    return scale

#cam
def channel_attention (input_tensor, ratio = 8, name = ""):
    input_shape = K.int_shape(input_tensor)

    shared_layer_one = tf.keras.layers.Dense(units = input_shape[-1]//ratio,
                                             activation = 'relu',
                                             kernel_initializer = 'he_normal',
                                             use_bias = False,
                                             bias_initializer='zeros',)
    shared_layer_two = tf.keras.layers.Dense(units = input_shape[-1],
                                             kernel_initializer = 'he_normal',
                                             use_bias = False,
                                             bias_initializer = 'zeros',)


    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(input_tensor)

    avg_pool = tf.reshape(avg_pool,[-1,1,input_shape[-1]])
    max_pool = tf.reshape(max_pool,[-1,1,input_shape[-1]])

    avg_pool = shared_layer_one(avg_pool)
    max_pool = shared_layer_one(max_pool)

    avg_pool1 = shared_layer_two(avg_pool)
    max_pool1 = shared_layer_two(max_pool)

# channel block attention model
    cbam_feature = tf.keras.layers.Add()([avg_pool1,max_pool1])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)
    return multiply([input_tensor,cbam_feature])

# spatial attention module
def spatial_attention(input_tensor, name=''):
    kernel_size = 3
    cbam_feature = input_tensor

    avg_pool = Lambda (lambda x: K.mean(x, axis = 2, keepdims = True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis = 2, keepdims= True))(cbam_feature)
    concat = Concatenate(axis = 2)([avg_pool,max_pool])

    cbam_feature = tf.keras.layers.Conv1D(filters = 1,
                                          kernel_size = kernel_size,
                                          strides = 1,
                                          padding = 'same',
                                          kernel_initializer = 'he_normal',
                                          use_bias = False,
                                          )(concat)
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return multiply([input_tensor,cbam_feature])

def cbam_block(cbam_feature, ratio = 8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

#tensorboard
tf.compat.v1.reset_default_graph()
logdir = 'D:\Jupyter\CNN\Test'

# CNN model creation
model = Sequential()
input_signal = tf.keras.Input(shape=(27,1))

#1
x1 = tf.keras.layers.Conv1D(filters = 16, kernel_size=32, strides=1 ,padding='same', activation='relu')(input_signal)
x2 = tf.keras.layers.BatchNormalization(epsilon=0.001)(x1)
x3 = tf.keras.layers.MaxPooling1D(pool_size=1,strides=1,padding='same')(x2)
z1 = cbam_block(x3,8)

#2
x4 = tf.keras.layers.Conv1D(filters = 32, kernel_size=32, strides=1 ,padding='same', activation='relu')(z1)
x5 = tf.keras.layers.BatchNormalization(epsilon=0.001)(x4)
x6 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=2,padding='same')(x5)
z2 = cbam_block(x6,8)

#3
# x7 = tf.keras.layers.Conv1D(filters = 64, kernel_size=32, strides=1 ,padding='same', activation='relu')(z2)
# x8 = tf.keras.layers.BatchNormalization(epsilon=0.001)(x7)
# x9 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=2,padding='same')(x8)
# z3 = cbam_block(x9,8)

# #4
# x10 = tf.keras.layers.Conv1D(filters = 128, kernel_size=32, strides=1 ,padding='same', activation='relu')(z3)
# x11 = tf.keras.layers.BatchNormalization(epsilon=0.001)(x10)
# x12 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=2,padding='same')(x11)
# z4 = cbam_block(x12,8)
#
# 5
# x13 = tf.keras.layers.Conv1D(filters = 256, kernel_size=32, strides=1 ,padding='same', activation='relu')(z4)
# x14 = tf.keras.layers.BatchNormalization(epsilon=0.001)(x13)
# x15 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=2,padding='same')(x14)
# z5 = cbam_block(x15,8)

# fully connected layer
x16 = tf.keras.layers.Flatten()(z2)
x17 = tf.keras.layers.Dense(64, activation = 'relu')(x16)
x18 = tf.keras.layers.Dense(3, activation = 'softmax')(x17)

# model execution
model = tf.keras.Model(inputs = input_signal, outputs = x18)
model.summary()
adam = Adam(learning_rate = 0.0001,beta_1 = 0.9, beta_2=0.999, epsilon = 1e-07, decay=0.0,amsgrad=False)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# model complexity
flops = get_flops(model, batch_size=16)
print("FLOPS: {flops / 10 ** 9:.03} G")

# display training history
train_history = model.fit(X_train,y_train,epochs = 25, batch_size=16,validation_split=0.2,shuffle=True)
history_dict = train_history.history
print(history_dict.keys())

writer = tf.summary.create_file_writer(logdir)
writer.close()

# test model
model.evaluate(X_test,y_test)

#save model
model.save('CNNCopy.h5')

# create ROC curve
ct = 1
list1 = []
for x in train_history.history['val_loss']:
    list1.append(ct)
    ct+=1
X_axis = list1
X,y1,y2,y3,y4 = X_axis, train_history.history['loss'], train_history.history['val_loss'], train_history.history['accuracy'], train_history.history['val_accuracy']
plt.xlabel('epoch')
plt.plot(X,y1,'#FF0000')
plt.plot(X,y2,'limegreen')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='lower left')
plt.twinx()
plt.plot(X,y3,'tab:orange')
plt.plot(X,y4,'dodgerblue')
plt.ylabel('Acc')
plt.title('Model')
plt.legend(['accuracy','val_accuracy'], loc='upper left')
plt.show()