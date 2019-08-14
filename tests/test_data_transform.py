import cv2
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))

from torchvision import transforms
from src.preprocess import Rescale, RandomCrop, Normalize, ToTensor
import cv2
import numpy as np
import torch

data_transform = transforms.Compose([  # Rescale(250),
    Rescale(100),
    # RandomCrop(224),
    RandomCrop(96),
    Normalize(),
    ToTensor()])


def test_transformers_on_obamas():
    # load in color image for face detection
    image = cv2.imread(os.path.join(HERE, '../images/obamas.jpg'))

    # switch red and blue color channels
    # --> by default OpenCV assumes BLUE comes first, not RED as in many images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(
        os.path.join(HERE, '../detector_architectures/haarcascade_frontalface_default.xml'))

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    image_copy = np.copy(image)
    fig = plt.figure(figsize=(20, 20))
    idx = 1
    rows = 1
    cols = len(faces)
    for (x,y,w,h) in faces:

        # Select the region of interest that is the face in the image
        roi = image_copy[y:y+h, x:x+w]

        ## TODO: Convert the face region from RGB to grayscale
        ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)

        transformed = data_transform({'image': roi, 'keypoints':  np.array([[0,0]])})
        transformed_image = transformed['image']

        ## TODO: Make facial keypoint predictions using your loaded, trained network
        ## perform a forward pass to get the predicted facial keypoints

        ## TODO: Display each detected face and the corresponding keypoints
        ax = fig.add_subplot(rows, cols, idx, xticks=[], yticks=[])
        #ax.imshow(transformed_image, cmap='gray')
        ax.imshow(transformed_image[0], cmap='gray')
        idx +=1

    plt.show()
