import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = self.load_image(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_image(self, image_path):
        # TODO: Implement with PIL (Phase 1)
        raise NotImplementedError

    def preprocess_image(self, image):
        # TODO: Implement transforms (Phase 1)
        raise NotImplementedError
