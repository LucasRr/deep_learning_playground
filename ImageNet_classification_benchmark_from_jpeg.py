'''

Benchmark pre-trained models on ImageNet validation set.
The data is loaded from jpegs. Use num_workers > 1 to avoid
any computational bottlenet dure to data loading/transforming.

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


if __name__ == "__main__":

	device = "mps"

	path_to_folder = "/Users/lucas/Documents/Data/ImageNet/ILSVRC2012_img_val/"
	path_to_labels_txt = "/Users/lucas/Documents/Data/ImageNet/imagenet_2012_validation_label_idxs.txt"

	classes = utils.data.get_ImageNet_classes()

	batch_size = 32
	num_workers = 4

	print(f"Load ImageNet data from jpegs")
	print(f"batch_size: {batch_size}, num_workers: {num_workers}")

	loss_fn = CrossEntropyLoss()


	alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
	alexnet_preprocessing = models.AlexNet_Weights.IMAGENET1K_V1.transforms()

	inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
	inception_preprocessing = models.Inception_V3_Weights.IMAGENET1K_V1.transforms()

	VGG16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
	VGG_preprocessing = models.VGG16_Weights.IMAGENET1K_V1.transforms()

	inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
	inception_preprocessing = models.Inception_V3_Weights.IMAGENET1K_V1.transforms()

	resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
	resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
	resnet_preprocessing = models.ResNet101_Weights.IMAGENET1K_V1.transforms()



	# data loaders with transforms:
	imagenet_alexnet = utils.data.ImageNet(path_to_folder,
											path_to_labels_txt,
											transform=alexnet_preprocessing)
	dataloader_alexnet = torch.utils.data.DataLoader(imagenet_alexnet,
										batch_size=batch_size,
										num_workers=num_workers)
	# -> used for alexnet, vgg and resnet

	imagenet_inception = utils.data.ImageNet(path_to_folder,
											path_to_labels_txt,
											transform=inception_preprocessing)
	dataloader_inception = torch.utils.data.DataLoader(imagenet_inception,
										batch_size=batch_size,
										num_workers=num_workers)


	print("\n=== AlexNet: ===")
	t = time.time()
	loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(alexnet.to(device), dataloader_alexnet, loss_fn, device)
	dt = time.time()-t
	print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
	print(f"Inference time: {dt/60:.2f} min")


	print("\n=== VGG16: ===")
	t = time.time()
	loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(VGG16.to(device), dataloader_alexnet, loss_fn, device)
	dt = time.time()-t
	print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
	print(f"Inference time: {dt/60:.2f} min")


	print("\n=== Inception: ===")
	t = time.time()
	loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(inception.to(device), dataloader_inception, loss_fn, device)
	dt = time.time()-t
	print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
	print(f"Inference time: {dt/60:.2f} min")


	print("\n=== ResNet18: ===")
	t = time.time()
	loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(resnet18.to(device), dataloader_alexnet, loss_fn, device)
	dt = time.time()-t
	print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
	print(f"Inference time: {dt/60:.2f} min")


	print("\n=== ResNet101: ===")
	t = time.time()
	loss, accuracy, top5_accuracy = utils.ML.evaluate_ImageNet(resnet101.to(device), dataloader_alexnet, loss_fn, device)
	dt = time.time()-t
	print(f"Loss: {loss:.3f}, accuracy: {accuracy:.3f}, top-5 accuracy: {top5_accuracy:.3f}")
	print(f"Inference time: {dt/60:.2f} min")

