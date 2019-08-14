from .context import *
from torchvision import transforms

HERE = os.path.dirname(os.path.realpath(__file__))


def test_model_with_sample_image():
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
    print(one_file_batch.shape)

    output = net(one_file_batch)
    assert output.shape == torch.Size([1, 136])


def test_model_with_naimesh_net_image__96_width():
    net = Net(image_width=96)
    sample = torch.rand(1, 1, 96, 96)
    output = net(sample)
    assert output.shape == torch.Size([1, 136])


def test_apply_first_conv_filter_to_image():
    net = Net()
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
    output = net.features[0](one_file_batch)
    print(output)
    print(output.shape)
