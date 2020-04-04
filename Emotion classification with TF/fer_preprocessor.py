# -*- coding: utf-8 -*-
import sys, os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

def save_photo(path, photo, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    Image.fromarray(photo).convert('RGB').save(os.path.join(path, file_name))
    print("wrote to {}".format(path))

def process_and_split_fer2013(DATA_PATH,SPLIT_PATH,CASCADE_PATH,RESOLUTION):
    """
    Reads the fer2013.csv file, preprocess faces and splits data into train 
    test and validation sets.
    
    Parameters:
    -------------------
    DATA_PATH: path to csv file
    SPLIT_PATH: destination path
    CASCADE_PATH: patho to haar cascade classifier
    RESOLUTION: resolution of fer images
    
    Returns:
    ----------
    None
    """
    width, height = RESOLUTION,RESOLUTION
    data = pd.read_csv(DATA_PATH)
    pixels = data['pixels'].tolist() 
    faces = []
    i = 0
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')] 
        face = np.asarray(face).reshape(width, height) 
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.array(faces, dtype='uint8')
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    i=0
    delete_indeces=list()
    # choose what faces to delete
    # for face in faces:
    #     maybe_face = face_cascade.detectMultiScale(face, 1.3, 4)
    #     if len(maybe_face) == 0:
    #         delete_indeces.append(i)
    #     i+=1
    #np.delete(faces,delete_indeces,axis=0) # delete potentional not_faces
    #print("/nIndicies that we deleted are: {}/n".format(delete_indeces))
    faces = np.expand_dims(faces, -1) 
    emotions = pd.get_dummies(data['emotion']).values

    X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)
    X_train = np.squeeze(X_train, 3)
    X_test = np.squeeze(X_test, 3)
    X_val = np.squeeze(X_val, 3)

    
    for i in range(np.shape(X_train)[0]):
        save_photo(os.path.join(SPLIT_PATH, "train", str(np.argmax(y_train[i]))), X_train[i], "img{}.png".format(i))
    for i in range(np.shape(X_test)[0]):
        save_photo(os.path.join(SPLIT_PATH, "test", str(np.argmax(y_test[i]))), X_test[i], "img{}.png".format(i))
    for i in range(np.shape(X_val)[0]):
        save_photo(os.path.join(SPLIT_PATH, "validation", str(np.argmax(y_val[i]))), X_val[i], "img{}.png".format(i))

    print("train set: shape = {}".format(np.shape(X_train)))
    print("test set: shape = {}".format(np.shape(X_test)))
    print("validation set: shape = {}".format(np.shape(X_val)))