import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from PIL import Image
import glob, os

# Default paths:
DATA_ROOT = "/Users/lucas/Documents/Data/"
IMAGENET_LABELS = "/Users/lucas/Documents/data/ImageNet/imagenet_labels.txt"

def get_MNIST_data_loaders(batch_size=64, path=DATA_ROOT):


    # Download train and test data:

    training_data_ = datasets.MNIST(
        root=path,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root=path,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # split training data into train and validation sets:
    train_split, val_split = torch.utils.data.random_split(
        training_data_, [50000, 10000],
        generator=torch.Generator().manual_seed(42))

    # Create data loaders.
    train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_split, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


class ImageNet(Dataset):
    
    def __init__(self, path_to_im_folder, path_to_labels_txt, transform):
        
        if not os.path.isdir(path_to_im_folder):
            raise FileNotFoundError(f"{path_to_im_folder} does not exist")

        self.path_to_im_folder = path_to_im_folder
        self.transform = transform
        
        # extract labels:
        with open(path_to_labels_txt, 'r') as f:
            txt = f.read()
            self.labels = txt.splitlines()
            
        self.num_images = len(glob.glob(os.path.join(path_to_im_folder, '*.JPEG')))
        
        if self.num_images != len(self.labels):
            raise FileNotFoundError(f"Found {self.num_images} .JPEG images but {len(self.labels)} labels")
            
    def __getitem__(self, idx):
        
        if idx >= self.num_images:
            raise StopIteration  # to stop iterator
        else:
            filename = f'ILSVRC2012_val_{idx+1:0>8}.JPEG'

            im_PIL = Image.open(os.path.join(self.path_to_im_folder, filename))

            if len(im_PIL.getbands()) != 3:
                # image is not RGB
                im_RGB = Image.new("RGB", im_PIL.size)
                im_RGB.paste(im_PIL)
            else:
                im_RGB = im_PIL

            try:
                im_preprocessed = self.transform(im_RGB)
            except:
                print(f"couldn't transform image {idx}")
                print(f"bands: {im_PIL.getbands()}")

            label = int(self.labels[idx])

            return im_preprocessed, label
    
    def __len__(self):
        return self.num_images


class ImageNet_preprocessed(Dataset):
    ''' 
        ImageNet validation dataset, preprocessed and
        stored as tensors. Pre-processed with:
            preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
    '''

    def __init__(self, path_to_im_folder, path_to_labels_txt):
        

        if not os.path.isdir(path_to_im_folder):
            raise FileNotFoundError(f"{path_to_im_folder} does not exist")

        self.path_to_im_folder = path_to_im_folder
        
        # extract labels:
        with open(path_to_labels_txt, 'r') as f:
            txt = f.read()
            self.labels = txt.splitlines()
            
        self.num_images = len(glob.glob(os.path.join(path_to_im_folder, '*.pt')))
        
        if self.num_images != len(self.labels):
            raise FileNotFoundError(f"Found {self.num_images} .pt images but {len(self.labels)} labels")
            
    def __getitem__(self, idx):
        
        if idx >= self.num_images:
            raise StopIteration  # to stop iterator
        else:
            filename = f'ILSVRC2012_val_{idx+1:0>8}.pt'

            im_tensor = torch.load(os.path.join(self.path_to_im_folder, filename))

            label = int(self.labels[idx])

            return im_tensor, label
    
    def __len__(self):
        return self.num_images


def get_ImageNet_classes(path_to_txt=IMAGENET_LABELS):
    ''' Return length-1000 list containing ImageNet classes '''
    with open(path_to_txt, 'r') as f:
        txt = f.read()
        classes = txt.splitlines()
    return classes


