'''

Benchmark pre-trained models on ImageNet validation set.
The data is loaded from pre-processed tensors to avoid any
computational bottleneck due to data loading/transforming.

Models compared:
- AlexNet
- VGG16
- Inception v3
- ResNet18
- ResNet101

'''

import torch
from torch.nn import CrossEntropyLoss
import torchvision
from torchvision import models
import torchvision.transforms as transforms


import numpy as np
import matplotlib.pyplot as plt
import json, os, glob, time

import utils
import utils.data, utils.ML, utils.models
from utils.models import number_of_parameters

device = "mps"

path_to_preprocessed_folder = "/Users/lucas/Documents/Data/ImageNet/ILSVRC2012_img_val_preprocessed/"
path_to_labels_txt = "/Users/lucas/Documents/Data/ImageNet/imagenet_2012_validation_label_idxs.txt"

imagenet_data = utils.data.ImageNet_preprocessed(path_to_preprocessed_folder, path_to_labels_txt)

classes = utils.data.get_ImageNet_classes()

batch_size = 32
num_workers = 0

print(f"Load ImageNet data from preprocessed tensors")
print(f"batch_size: {batch_size}, num_workers: {num_workers}")

dataloader = torch.utils.data.DataLoader(imagenet_data,
									batch_size=batch_size,
									num_workers=num_workers)

loss_fn = CrossEntropyLoss()


alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
AlexNet_preprocessing = models.AlexNet_Weights.IMAGENET1K_V1.transforms()

inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
inception_preprocessing = models.Inception_V3_Weights.IMAGENET1K_V1.transforms()

VGG16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
VGG_preprocessing = models.VGG16_Weights.IMAGENET1K_V1.transforms()

inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
inception_preprocessing = models.Inception_V3_Weights.IMAGENET1K_V1.transforms()

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
resnet_preprocessing = models.ResNet101_Weights.IMAGENET1K_V1.transforms()


print("\n=== AlexNet: ===")
t = time.time()
loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(alexnet.to(device), dataloader, loss_fn, device)
dt = time.time()-t
print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
print(f"Inference time: {dt/60:.2f} min")


print("\n=== VGG16: ===")
t = time.time()
loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(VGG16.to(device), dataloader, loss_fn, device)
dt = time.time()-t
print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
print(f"Inference time: {dt/60:.2f} min")


print("\n=== Inception: ===")
print("(Note: Inception uses a different preprocessing step which might affect the performance)")
t = time.time()
loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(inception.to(device), dataloader, loss_fn, device)
dt = time.time()-t
print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
print(f"Inference time: {dt/60:.2f} min")


print("\n=== ResNet18: ===")
t = time.time()
loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(resnet18.to(device), dataloader, loss_fn, device)
dt = time.time()-t
print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
print(f"Inference time: {dt/60:.2f} min")


print("\n=== ResNet101: ===")
t = time.time()
loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(resnet101.to(device), dataloader, loss_fn, device)
dt = time.time()-t
print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
print(f"Inference time: {dt/60:.2f} min")

