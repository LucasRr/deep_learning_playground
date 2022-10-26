This repo is a playground to implement, train, and benchmark various deep learning models on a variety of tasks. All models are implemented in Pytorch. 



Example notebooks:

- `MNIST_classification_MLP.ipynb`: test various multi-layer perceptron architectures for MNIST classification
- `MNIST_classification_CNN.ipynb`: test various CNN architectures for MNIST classification
- `CIFAR10_classification.ipynb`: implement and train (from scratch) various popular CNN-based architectures (LeNet5, AlexNet, VGG, Inception, ResNet) for CIFAR10 classification 
- `CNN_for_image_denoising.ipynb`: compare filtering with various fixed or learned convolution kernel, and simple mutlilayer CNNs for image denoising. Evaluated on MNIST
- `ImageNet_classification_pretrained.ipynb`: benchmark various pretrained models (available from Pytorch's `torchvision`) for ImageNet classification, in terms of accuracy vs. inference speed.
- `transfer_learning.ipynb`: use transfer learning to classify CIFAR10 images using models pre-trained on MNIST

