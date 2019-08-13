from torchvision import transforms

from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_transformers():
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
