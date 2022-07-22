import numpy as np
from keras.models import *
import os
import pandas as pd

# CPU selection
os.environ['CUDA_DEVICE_ORDER']= '[PCI_BUS_ID]'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

# Selectively test model
my_model=load_model('D:\Jupyter\CNN\Test\Saved_Models\CNNCopy.h5')
X = np.load('D:\Jupyter\CNN\Test\Data_NPY\cloth.test.npy',allow_pickle=True)
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

#Save prob
pd.DataFrame(pre_proba).to_csv("D:\Jupyter\CNN\Test\Output\CNN_Prob.csv")

