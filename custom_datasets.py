import os
from torch.utils.data import Dataset
from PIL import Image

class TestDataset(Dataset):
    """
    A custom dataset class for test images in PyTorch, suitable for situations where the test set does not include labels.

    This dataset class is designed to handle a directory of images, applying specified transformations to these images and returning both the transformed image and its file name. It is particularly useful for inference tasks where the labels are not available, and the image file names are required for identification or submission purposes.

    Parameters:
    - data_path (str): The file path to the directory containing the test images.
    - transforms (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.Compose` transformations. If `None`, no transformation is applied.

    The class overrides the `__getitem__` method of the PyTorch `Dataset` class to return a tuple containing the transformed image and its file name. It also overrides the `__len__` method to return the total number of images in the dataset.

    Example usage:
        # Assuming images are located in 'path/to/test/images' and we have some transformations
        test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        test_dataset = TestDataset(data_path='path/to/test/images', transforms=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    Note: This class assumes that the images are in a format compatible with `PIL.Image.open` and that they are stored in a flat directory structure.
    """
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.images = os.listdir(data_path)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.data_path, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, image_name

    def __len__(self):
        return len(self.images)


class WrapperDataset(Dataset):
    """
    A wrapper dataset class in PyTorch for applying different transformations to subsets of a dataset, typically used to handle train and validation splits.

    This class is designed to wrap around a given dataset and apply specific transformations only to selected indices. This is useful when the same dataset is split into training and validation sets, and different transformations are needed for each set (e.g., augmentation for training data but not for validation data).

    Parameters:
    - dataset (Dataset): The original dataset to be wrapped. This dataset should implement the standard PyTorch Dataset methods.
    - indices (list of int): A list of indices indicating which subset of the dataset to use. This allows the wrapper to focus on either the training or validation set.
    - transforms (callable, optional): A function/transform that takes in an image and returns a transformed version. E.g., `transforms.Compose` transformations for the specified subset. If `None`, no transformation is applied.

    The class overrides the `__getitem__` and `__len__` methods of the PyTorch `Dataset` class. In the `__getitem__` method, the transformation is applied to the data points (if specified) before they are returned.

    Example usage:
        # Assume `full_dataset` is a dataset object and `train_indices`, `val_indices` are lists of indices
        train_transforms = transforms.Compose([...]) # some transformations
        val_transforms = transforms.Compose([...]) # different or no transformations

        train_dataset = WrapperDataset(full_dataset, train_indices, train_transforms)
        val_dataset = WrapperDataset(full_dataset, val_indices, val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    Note: It's important to ensure that the `indices` provided correspond correctly to the desired train/validation splits from the `dataset`.
    """
    def __init__(self, dataset, indices, transforms=None):
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms
    
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.indices)