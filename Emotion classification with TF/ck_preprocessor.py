import os
import numpy as np
import cv2


class Preprocessor:
    def __init__(self,CASCADE_PATH,DEST_PATH,RESOLUTION):
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.dest_path = DEST_PATH
        self.resolution = RESOLUTION
    def process_data(self, label_path, feature_path):   
        """
        Extracts the face from the images and saves the images in the DEST_PATH.
        From one sequence(expressions for one emotion for one person) of images it
        only takes first image and the last 3, because images in between are not 
        recognizible emotion
        
        Parameters
        -----
        label_path : path to the folder with labels of emotions
    
        feature_path : path to the folder with images
    
        Returns
        -----
        None
        """
        i = 0
        for root, dirs, files in os.walk(label_path, topdown=False):
            if not files:
                continue
            for name in files:
                
                file = open(os.path.join(root, name), 'r')
                a = int(float(file.read().strip())) # read the label of the image
                root_path = root.replace('labels', 'images')
                img_files = os.listdir(root_path)
                num_files = len(img_files)
                j = 0
                for file in img_files:
                    if j == 0:
                        self.crop_save_face(os.path.join(root_path, file), self.dest_path + '\\0\\', file)
                        print('Processed image {}'.format(i))
                        i+=1
                    elif j >= num_files - 3: # take only the last 3 pictures, because other pictures are not recognizible emotions
                        self.crop_save_face(os.path.join(root_path, file), self.dest_path + '\\{}\\'.format(a), file)
                        print('Processed image {}'.format(i))
                        i+=1
                    j += 1

    def crop_save_face(self, image_path, destination_path, destination_file):
        img = cv2.imread(image_path)
        if img is None:
            print(image_path)
            print("Can't open image file")
            return 0

        faces = self.face_cascade.detectMultiScale(img, 1.1, minSize=(250, 250))
        if faces is None:
            print('Failed to detect face')
            return 0

        for (x, y, w, h) in faces:
            faceimg = img[y:y + w, x:x + h]
            lastimg = cv2.resize(faceimg, (self.resolution, self.resolution))
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            cv2.imwrite(destination_path + "\\{}".format(destination_file), lastimg)
            