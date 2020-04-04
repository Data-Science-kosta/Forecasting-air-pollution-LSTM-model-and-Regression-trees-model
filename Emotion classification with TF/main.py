from ck_preprocessor import Preprocessor
from splitter import split_ck
from fer_preprocessor import process_and_split_fer2013
from data_loader import Loader

from Model import Model
from Trainer import Trainer # trains the model on a single dataset
from Trainer1 import Trainer1 # trains the model on 2 datasets
from Predictor import Predictor

from live_stream import *

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
#-----------------------------------------------------------------------------
# To log to tensorboard type: "tensorboard --logdir=mainLogs --host=127.0.0.1"
# After that copy and paste the link in the browser, make sure it is
# localhost:6006(port)
#-----------------------------------------------------------------------------
HAAR_CASCADE_PATH = ".\\haarcascade_frontalface_default.xml" # path to the haar cascade classifier
# paths for CK dataset
DEST_PATH = '.\\data\\processed\\CK48' # path where you want to place processed CK images
LABELS_PATH = '.\\data\\CK\\labels' # path to the CK labels 
IMAGES_PATH = '.\\data\\CK\\images' # path to the CK images
SPLIT_PATH = '.\\data\\splitted\\CK48' # path where you want to plave splitted CK images
# paths for FER dataset
DATA_PATH_fer2013 = '.\\data\\fer2013\\fer2013.csv' # path to the FER csv file
SPLIT_PATH_fer2013 = '.\\data\\splitted\\fer2013' # path where you want to plave splitted FER images

CK_LABELS = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
FER_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

TRAINED_WEIGHTS_PATH = '.\\model\\trained' # path to the pretrained weights

RESOLUTION = 48 # 48
RESOLUTION_fer = 48 # resolution for FER dataset must be 4
learning_rate=0.001
epochs=40
batch_size=32

        
if __name__ == '__main__':
    while(True):
        print("-------------------------------------------------------------------------------------")
        print('Choose "1" to extract the faces from images and to drop images with unclear emotion (CK dataset)')
        print('Choose "2" to split the data into the train,test and validation (CK dataset')
        print('Choose "3" to process and split fer 2013 dataset')
        print('Choose "4" to load the CK dataset')
        print('Choose "5" to load the FER dataset')
        print('Choose "6" to start training')
        print('Choose "7" to load both datasets and to start training')
        print('Choose "8" to load pretrained weights, and test them on the desktop camera')
        print('Choose "e" to exit')
        print("\nYour choice?: ")
        inp = input()
        if inp == '1': # preprocess CK images
            p = Preprocessor(HAAR_CASCADE_PATH, DEST_PATH, RESOLUTION) 
            p.process_data(LABELS_PATH, IMAGES_PATH) 
        elif inp == '2': # split CK dataset on train,test and validation sets
            split_ck(DEST_PATH,SPLIT_PATH,RESOLUTION) 
        elif inp == '3': # preprocess and split FER 2013 dataset
            process_and_split_fer2013(DATA_PATH_fer2013,SPLIT_PATH_fer2013,HAAR_CASCADE_PATH,RESOLUTION_fer)
        elif inp == '4': # load CK dataset
            l = Loader(RESOLUTION) 
            labels_train, labels_valid, labels_test, images_train, images_valid, images_test = l.load_data(SPLIT_PATH)
        elif inp == '5': # load FER 2013 dataset
            l = Loader(RESOLUTION_fer) 
            labels_train, labels_valid, labels_test, images_train, images_valid, images_test = l.load_data(SPLIT_PATH_fer2013)
        elif inp == '6': # train and test
            #shuffling the data
            indices1 = np.arange(labels_train.shape[0])
            np.random.shuffle(indices1)
            labels_train=labels_train[indices1]
            images_train=images_train[indices1]
            indices2 = np.arange(labels_test.shape[0])
            np.random.shuffle(indices2)
            labels_test=labels_test[indices2]
            images_test=images_test[indices2]
            indices3 = np.arange(labels_valid.shape[0])
            np.random.shuffle(indices3)
            labels_valid=labels_valid[indices3]
            images_valid=images_valid[indices3]
            # scaling the images
            images_train = images_train / 255.0
            images_valid = images_valid / 255.0
            images_test = images_test / 255.0
            # creating the model
            model=Model(RESOLUTION,learning_rate) # be careful when choosing RESOLUTION
            trainer=Trainer(images_train, labels_train, images_valid, labels_valid, model, epochs, batch_size)
            # training
            trainer.train()
            # testing
            print('\n\nTesting started\n\n')
            print(images_test.shape)
            _, _, matrix = trainer.validate(images_test, labels_test, images_test.shape[0]) 
            
            print(matrix)
            maximum=max(max(x) for x in matrix)
            matrix=matrix*255.0/maximum
            plt.imshow(matrix, cmap='Blues')
            plt.show()
            trainer.save('.\\model\\trained2')
        elif inp == '7':
            l = Loader(RESOLUTION_fer) 
            labels_train, labels_valid, labels_test, images_train, images_valid, images_test = l.load_data(SPLIT_PATH)
            labels_train1, labels_valid1, labels_test1, images_train1, images_valid1, images_test1 = l.load_data(SPLIT_PATH_fer2013)
            #shuffling the data
            indices1 = np.arange(labels_train.shape[0])
            np.random.shuffle(indices1)
            labels_train=labels_train[indices1]
            images_train=images_train[indices1]
            indices2 = np.arange(labels_test.shape[0])
            np.random.shuffle(indices2)
            labels_test=labels_test[indices2]
            images_test=images_test[indices2]
            indices3 = np.arange(labels_valid.shape[0])
            np.random.shuffle(indices3)
            labels_valid=labels_valid[indices3]
            images_valid=images_valid[indices3]
            # second dataset
            indices1 = np.arange(labels_train1.shape[0])
            np.random.shuffle(indices1)
            labels_train1=labels_train1[indices1]
            images_train1=images_train1[indices1]
            indices2 = np.arange(labels_test1.shape[0])
            np.random.shuffle(indices2)
            labels_test1=labels_test1[indices2]
            images_test1=images_test1[indices2]
            indices3 = np.arange(labels_valid1.shape[0])
            np.random.shuffle(indices3)
            labels_valid1=labels_valid1[indices3]
            images_valid1=images_valid1[indices3]
            # scaling the images
            images_train = images_train / 255.0
            images_valid = images_valid / 255.0
            images_test = images_test / 255.0
            images_train1 = images_train1 / 255.0
            images_valid1 = images_valid1 / 255.0
            images_test1 = images_test1 / 255.0
            # creating the model
            model=Model(RESOLUTION_fer,learning_rate) # be careful when choosing RESOLUTION
            trainer=Trainer1(images_train,labels_train,images_valid,labels_valid,images_train1,labels_train1,images_valid1,labels_valid1,model,epochs,batch_size)
            # training
            trainer.train()
            # testing
            print('\n\nTesting started\n\n')
            print(images_test.shape)
            _, _, matrix = trainer.validate(images_test, labels_test, images_test.shape[0]) 
            
            print(matrix)
            maximum=max(max(x) for x in matrix)
            matrix=matrix*255.0/maximum
            plt.imshow(matrix, cmap='Blues')
            plt.show()
            trainer.save('.\\model\\trained3')
        elif inp == '8':
            model = Model(RESOLUTION_fer)
            predictor = Predictor(model,RESOLUTION_fer)
            predictor.restore(TRAINED_WEIGHTS_PATH)
            live_stream(predictor,CK_LABELS,HAAR_CASCADE_PATH)
        elif inp == 'e':
            break
        else:
            print("###Wrong arguments###")