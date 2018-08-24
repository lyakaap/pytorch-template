import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import sampler


class CSVDataset(Dataset):

    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = cv2.imread(row['ImageID'])
        target = row['class']
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.df)


class TestDataset(Dataset):

    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img = cv2.imread(str(self.img_paths[index]))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)


class InfiniteSampler(sampler.Sampler):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return None
