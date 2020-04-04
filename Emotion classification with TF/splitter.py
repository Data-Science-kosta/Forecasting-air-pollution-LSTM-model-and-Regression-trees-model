import os
import numpy as np
from PIL import Image
from sklearn import model_selection as ms

def split_ck(PROCESSED_PATH,SPLIT_PATH,RESOLUTION):
    """
    Splits the processed data into train,test and validation sets.
    From every label is taken a same percentage of images.
    
    Parameters
    -----
    PROCESSED_PATH: path to the preprocessed images

    RESOLUTION: resolution of images in PROCESSED_PATH

    Returns
    -----
    None
    """
    for dirs in os.listdir(PROCESSED_PATH): 
        imgs_list = os.listdir(os.path.join(PROCESSED_PATH, dirs))
        num_imgs = len(imgs_list)
        temp_matrix = np.empty((num_imgs, RESOLUTION, RESOLUTION))
        j = 0
        for img in imgs_list:
            im = Image.open(os.path.join(PROCESSED_PATH, dirs, img)).convert('F')
            temp_matrix[j] = np.array(im)
            j+=1
        label_matrix = np.ones(num_imgs) * int(dirs)
        x_train, x_test, y_train, y_test = ms.train_test_split(temp_matrix, label_matrix, test_size=0.2, random_state=1)
        x_train, x_validation, y_train, y_validation = ms.train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        for i in range(np.shape(x_train)[0]):
            save_photo(os.path.join(SPLIT_PATH, "train", dirs), x_train[i], "img{}.png".format(i))
        for i in range(np.shape(x_test)[0]):
            save_photo(os.path.join(SPLIT_PATH, "test", dirs), x_test[i], "img{}.png".format(i))
        for i in range(np.shape(x_validation)[0]):
            save_photo(os.path.join(SPLIT_PATH, "validation", dirs), x_validation[i], "img{}.png".format(i))

def save_photo(path, photo, file_name):
    """
    Saves the photo to the path. If path does not exist it creates it
    """
    if not os.path.exists(path):
        os.makedirs(path)
    Image.fromarray(photo).convert('RGB').save(os.path.join(path, file_name))
    print("wrote to {}".format(path))