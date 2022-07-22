import numpy as np
from keras.models import *
import os

# CPU selection
os.environ['CUDA_DEVICE_ORDER']= '[PCI_BUS_ID]'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

# Selectively test model
my_model=load_model('CNNCopy.h5')
X = np.load('cloth.test.npy',allow_pickle=True)
prediction = np.argmax(my_model.predict(X),axis=-1)
pre_proba = my_model.predict(X)
print(pre_proba)
print(prediction)
for x in prediction:
    if x == 0:
        print('table')
    if x == 1:
        print('cloth')
    if x == 2:
        print('Aluminum')