import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
'''

Preprocess ImageNet images and save them as .pt tensors

'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json, os, glob, time

from utils.data import ImageNet


preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
)])



if __name__ == "__main__":

    path_to_im_folder = "/Users/lucas/Documents/Data/ImageNet/ILSVRC2012_img_val/"
    path_to_labels_txt = "/Users/lucas/Documents/Data/ImageNet/imagenet_2012_validation_label_idxs.txt"

    path_to_output = "/Users/lucas/Documents/Data/ImageNet/ILSVRC2012_img_val_preprocessed/"

    imagenet_val = ImageNet(path_to_im_folder, path_to_labels_txt, preprocess)

    for idx, (im, _) in enumerate(imagenet_val):

        filename = f'ILSVRC2012_val_{idx+1:0>8}.pt'

        torch.save(im, os.path.join(path_to_output, filename))


