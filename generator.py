S# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:26:55 2019

@author: uku4kor
"""

import numpy as np
import pandas as pd
import os
import re
#from keras.preprocessing import image

def generator(csv_path, batch_size, img_height, img_width, channels, augment=False):

    ########################################################################
    # The code for parsing the CSV (or loading the data files) should goes here
    # We assume there should be two arrays after this:
    #   img_path --> contains the path of images
    #   annotations ---> contains the parsed annotaions
    ########################################################################
    
    n_samples = len(next(os.walk(csv_path))[2])
    batch_img = np.zeros((batch_size, img_height, img_width, channels))
    idx = 0
    while True:
        batch_img_path = os.listdir(csv_path)[idx:idx+batch_size]
        batch_target = []
        target1 = []
        target2 = []
        target3 = []
        target4 = []
        
        for i, p in zip(range(batch_size), batch_img_path):
            
            path =  os.path.join(csv_path,p)
            dat = pd.read_csv(path)           
            batch_img[i,:,0,0] = dat.iloc[:,0]
            batch_img[i,:,0,1] = dat.iloc[:,1]
            batch_img[i,:,0,2] = dat.iloc[:,2]
            batch_img[i,:,0,3] = dat.iloc[:,3]
            batch_img[i,:,0,4] = dat.iloc[:,4]
            
            if(bool(re.match("(.*car.*)",p))):
                target1.append(1)
            else:
                target1.append(0)
                #target = 1
            
            if(bool(re.match(".*TwoWheeler.*",p))):
                target2.append(1)
            else:
                target2.append(0)
                #target = 2
                
            if(bool(re.match(".*Pedestrian.*",p))):
                target3.append(1)
            else:
                target3.append(0)
                #target = 3
                
            if(bool(re.match(".*Other.*",p))):
                target4.append(1)
            else:
                target4.append(0)
                #target = 4
           
        batch_target = np.column_stack((target1,target2))
        batch_target = np.column_stack((batch_target,target3))
        batch_target = np.column_stack((batch_target,target4))
        

#        if augment:
            ############################################################
            # Here you can feed the batch_img to an instance of 
            # ImageDataGenerator if you would like to augment the images.
            # Note that you need to generate images using that instance as well
            ############################################################
        
        
        idx += batch_size
        if idx > n_samples - batch_size:
            idx = 0

        yield batch_img, batch_target
        


train_gen = generator('D:/Sudhu/Project/data_train_test_shuffle/Train', 32, 64, 1, 5)
val_gen = generator('D:/Sudhu/Project/data_train_test_shuffle/Validation', 32, 64, 1, 5)
test_gen = generator('D:/Sudhu/Project/data_train_test_shuffle/Test1', 164, 64, 1, 5)
test_gen2 = generator('D:/Sudhu/Project/data_train_test_shuffle/Test2', 52, 64, 1, 5)
test_gen3 = generator('D:/Sudhu/Project/data_train_test_shuffle/Test3', 54, 64, 1, 5)
test_gen4 = generator('D:/Sudhu/Project/data_train_test_shuffle/Test4', 26, 64, 1, 5)
test_gen_fake = generator('D:/Sudhu/Project/data_train_test_shuffle/Test', 30, 64, 1, 5)

#
#model.fit_generator(train_gen,
#                    steps_per_epoch= , # should be set to num_train_samples / train_batch_size 
#                    epochs= , # your desired number of epochs,
#                    validation_data= val_gen,
#                    validation_steps= # should be set to num_val_samples / val_batch_size)
                    

from keras.models import Sequential
#from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.callbacks import CSVLogger


input_shape = (64, 1, 5)                    
model = Sequential()

model.add(Conv2D(32, (3, 1), input_shape=input_shape))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 1), activation='relu'))
model.add(Conv2D(64, (3, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Activation("relu"))

model.add(Dense(4))
model.add(Activation("softmax"))

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])
                    
model.fit_generator(
    train_gen,
    steps_per_epoch=32 ,
    epochs=250,
    validation_data=val_gen,
    validation_steps=8)                  


#y = model.predict_generator(test_gen,steps = 10)

y = model.predict_generator(test_gen,steps = 1)
y2 = model.predict_generator(test_gen2,steps = 1)
y3 = model.predict_generator(test_gen3,steps = 1)
y4 = model.predict_generator(test_gen4,steps = 1)
y_fake = model.predict_generator(test_gen_fake,steps = 1)
