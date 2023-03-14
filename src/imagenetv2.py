from PIL import Image
import os
import pathlib
from torch.utils.data import Dataset

from src.imagenet_classnames import get_classnames


class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai',
                 distributed=False):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.distributed = distributed

    def name(self):
        return 'imagenet'


class ImageNetV2Dataset(Dataset):
    def __init__(self, transform=None, location="."):
        self.dataset_root = pathlib.Path(location)
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ImageNetV2(ImageNet):

    def get_test_dataset(self, subdir='imagenet-v2/imagenetv2-matched-frequency-format-val/'):

        valdir = os.path.join(self.location, subdir)
        return ImageNetV2Dataset(transform=self.preprocess, location=valdir)

