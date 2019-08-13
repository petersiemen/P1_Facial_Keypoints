from .context import *
from torchvision import transforms
HERE = os.path.dirname(os.path.realpath(__file__))


def test_model():
    net = Net()
    print(net)


    data_transform = transforms.Compose([Rescale(250),
                                         RandomCrop(224),
                                         Normalize(),
                                         ToTensor()])

    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file=os.path.join(HERE, '../data/training_frames_keypoints.csv'),
                                                 root_dir=os.path.join(HERE, '../data/training/'),
                                                 transform=data_transform)

    sample = transformed_dataset[0]

    one_file_batch = sample['image'].unsqueeze(0).float()

    output = net(one_file_batch)
    print(output)
