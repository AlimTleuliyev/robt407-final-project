import os
from torch.utils.data import Dataset
from PIL import Image

class TestDataset(Dataset):
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