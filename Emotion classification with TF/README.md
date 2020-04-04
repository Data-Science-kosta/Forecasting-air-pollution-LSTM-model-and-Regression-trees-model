# Emotion Classification with TensorFlow
##### This is a fully convolutional neural network that classifies 8 different emotions from a given images.
##### The program works in real time through the desktop camera.
### DATA:
For training this network 2 datsets were used:

1: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

2: https://www.pitt.edu/~emotion/ck-spread.htm
### TRAINING:
The network was first trained on the CK dataset. Images in that set are all taken in the perfect environment, camera position, lights, contrast... On that dataset model easily achieves 99% accuracy on the test set, but when you test it in the real environment (through the desktop camera) it fails to predict half of the emotions accurately. That was espected because distribution of the data used for training is way too different from the distribution used for testing. One more downside of this dataset is that it is quite small. So the next step was to get more data which has similar distribution as the data in the real environment. So FER datset was joined with CK dataset and the model was trained. The accuracy improved greatly, but unfortunately not enough because the FER datset contains a lot of wrongly labeled examples and a lot of images which are not even a faces. 
Next steps would be creating my own dataset that consits of images taken in the real environment and retraining the network.
### NOTE:
Datasets and pretrained network weights were to big to push them on the Github.
