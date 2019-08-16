from torchvision import transforms
import math
import pytest

from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_normalize_and_rescale_keypoints():
    df = pd.read_csv(os.path.join(HERE, '../data/training_frames_keypoints.csv'))
    all = df.to_numpy()
    data = all[:, 1:]
    pairs = data.reshape(-1, 2)
    print("\nMean of keypoints x-values: {}".format(pairs[:, 0].mean()))
    print("Mean of keypoints y-values: {}".format(pairs[:, 1].mean()))
    print("Standard deviation of keypoints x-values: {}".format(pairs[:, 0].std()))
    print("Standard deviation of keypoints y-values: {}".format(pairs[:, 1].std()))


def test_rescale():
    data_transform = transforms.Compose([Rescale(100)])

    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file=os.path.join(HERE, '../data/training_frames_keypoints.csv'),
                                                 root_dir=os.path.join(HERE, '../data/training/'),
                                                 transform=data_transform)

    n = len(transformed_dataset)
    for i in range(n):
        sample = transformed_dataset[i]
        # numpy image: H x W x C
        w, h, c = sample['image'].shape
        assert min(w, h) == 100


def test_all_transformers():
    data_transform = transforms.Compose([Rescale(250),
                                         RandomCrop(224),
                                         Normalize(),
                                         ToTensor()])

    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file=os.path.join(HERE, '../data/training_frames_keypoints.csv'),
                                                 root_dir=os.path.join(HERE, '../data/training/'),
                                                 transform=data_transform)

    print('Number of images: ', len(transformed_dataset))

    # make sure the sample tensors are the expected size
    for i in range(5):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size())
