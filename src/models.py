## TODO: define the convolutional neural network architecture

# can use the below import should you choose to initialize the weights of your Net
import math
import torch.nn as nn


class Net(nn.Module):
    """
    This network implements the architecture from the NaimeshNet https://arxiv.org/pdf/1710.00977.pdf
    with the difference of predicting 136 facial key points instead of only 2 as in the paper.


    Layer   NumberLayer     Shape
    1       Input1          (1,96,96)
    2       Convolution2d1  (32,93,93)
    3       Activation1     (32,93,93)
    4       Maxpooling2d1   (32,46,46)
    5       Dropout1        (32,46,46)

    6       Convolution2d2  (64,44,44)
    7       Activation2     (64,44,44)
    8       Maxpooling2d2   (64,22,22)
    9       Dropout2        (64,22,22)
    10      Convolution2d3  (128,21,21)
    11      Activation3     (128,21,21)
    12      Maxpooling2d3   (128,10,10)
    13      Dropout3        (128,10,10)
    14      Convolution2d4  (256,10,10)
    15      Activation4     (256,10,10)
    16      Maxpooling2d4   (256,5,5)
    17      Dropout4        (256,5,5)

    18      Flatten1        (6400)
    19      Dense1          (1000)
    20      Activation5     (1000)
    21      Dropout5        (1000)
    22      Dense2          (1000)
    23      Activation6     (1000)
    24      Dropout6        (1000)
    25      Dense3          (2)

    """

    def __init__(self, image_width=224):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        self.features = nn.Sequential(
            # input 1x224x224
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4)),  # 32x221x221
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # 32x110x110
            nn.Dropout2d(p=0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),  # 64x108x108
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # 64x54x54
            nn.Dropout2d(p=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),  # 128x53x53
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # 128x26x26
            nn.Dropout2d(p=0.3),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),  # 256x26x26
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # 256x13x13
            nn.Dropout2d(p=0.4)
        )

        def output_image_width_conv(image_width, kernel_size, stride=1, padding=0):
            return math.floor((image_width - kernel_size + 2 * padding) / stride + 1)

        image_width_for_classifier = output_image_width_conv(image_width=image_width, kernel_size=4)
        image_width_for_classifier = math.floor(image_width_for_classifier / 2)
        image_width_for_classifier = output_image_width_conv(image_width=image_width_for_classifier, kernel_size=3)
        image_width_for_classifier = math.floor(image_width_for_classifier / 2)
        image_width_for_classifier = output_image_width_conv(image_width=image_width_for_classifier, kernel_size=2)
        image_width_for_classifier = math.floor(image_width_for_classifier / 2)
        image_width_for_classifier = output_image_width_conv(image_width=image_width_for_classifier, kernel_size=1)
        image_width_for_classifier = math.floor(image_width_for_classifier / 2)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256 * image_width_for_classifier * image_width_for_classifier, out_features=1000),
            nn.ELU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=1000, out_features=1000),
            nn.Dropout2d(p=0.6),
            nn.Linear(in_features=1000, out_features=136),

        )

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten feature maps
        x = self.classifier(x)
        return x
