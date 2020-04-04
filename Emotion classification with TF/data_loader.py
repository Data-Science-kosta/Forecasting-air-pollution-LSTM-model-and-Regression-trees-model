import numpy as np
import os
from PIL import Image

class Loader:
    def __init__(self,RESOLUTION):
        self.resolution = RESOLUTION
    def load_data(self, path):
        """
        Loads the splitted data from the path

        Parameters
        ----------
        path : Path to the splitted dataset (SPLIT_PATH)
        
        Returns
        -------
        train_labels
        validation_labels
        test_labels
        train_images
        validation_images
        test_images
        """
        self.train_labels = np.empty(0)
        self.test_labels = np.empty(0)
        self.validation_labels = np.empty(0)
        self.train_images = np.empty((0, self.resolution, self.resolution))
        self.validation_images = np.empty((0, self.resolution, self.resolution))
        self.test_images = np.empty((0, self.resolution, self.resolution))
        lst1=[]
        lst2=[]
        for dirs in os.listdir(path):
            for label in os.listdir(os.path.join(path, dirs)):
                label_path = os.path.join(path, dirs, label)
                for file in os.listdir(label_path):
                    image=Image.open(os.path.join(label_path,file)).convert('F')
                    lst1.append(np.array(image))
                    lst2.append(label)
            if dirs == "train":
                self.train_images=np.array(lst1)
                self.train_labels=np.array(lst2)
                lst1.clear()
                lst2.clear()
            if dirs == "test":
                self.test_images=np.array(lst1)
                self.test_labels=np.array(lst2)
                lst1.clear()
                lst2.clear()
            if dirs == "validation":
                self.validation_images=np.array(lst1)
                self.validation_labels=np.array(lst2)
                lst1.clear()
                lst2.clear()
        print("LOADED")
        return self.train_labels, self.validation_labels, self.test_labels, self.train_images, self.validation_images, self.test_images