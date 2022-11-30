This repo is a playground to experiment with various deep learning models, on a variety of tasks, such as image classification, image denoising, dimensionality reduction, transfer learning etc. All models are implemented in Pytorch. 

Most experiments can be found as a collection of notebooks.

##### List of notebooks:

- `MNIST_classification_MLP.ipynb`: multi-layer perceptron architectures for MNIST classification
- `MNIST_classification_CNN.ipynb`: CNN architectures for MNIST classification
- `CIFAR10_classification.ipynb`: popular CNN-based architectures (LeNet5, AlexNet, VGG, Inception, ResNet) for CIFAR10 classification (trained from scratch)
- `CNN_for_image_denoising.ipynb`: compare filtering with various fixed or learned convolution kernels, and simple mutlilayer CNNs for image denoising. Evaluated on MNIST
- `ImageNet_classification_pretrained.ipynb`: benchmark various pretrained models (available from Pytorch's `torchvision`) for ImageNet classification, in terms of accuracy vs. inference speed.
- `transfer_learning.ipynb`: transfer learning to classify CIFAR10 images using models pre-trained on ImageNet
- `network_visualization_activation_maximization.ipynb`: visualize convolution filters of popular CNN models, perform activation maximization for network visualization, and experiment with adversarial examples. 
- `autoencoders_MNIST.ipynb`: compare linear, fully-connected and convolutional autoencoders on MNIST
- `autoencoders_face_images.ipynb`: experiment with autoencoders for face representation, generation and reconstruction


---


Below are some benchmark results:

##### CIFAR-10 classification:

A comparison of popular architectures on the CIFAR-10 dataset. Note that those are my own implementations, which have been adapted (and simplified) to take 32x32 images as input, and trained from scratch. The results might thus not be optimal and reflect the results demonstrated on ImageNet.

| Model | Test accuracy (%) | Training time (per epoch)|
| --- | --- | --- |
| Fully-connected | 0.50 | 5.3s|
| LeNet/AlexNet-style network | 0.66 | 5.9s |
| VGG-style network | 0.71 | 10.1s|
| Inception-style network | 0.76 | 15.6s |
| ResNet-style network | 0.66 | 12.1s |
| Transfer learning (ImageNet-AlexNet features) | 0.83 | 92.5s |
| Transfer learning (ImageNet-ResNet18 features) | 0.78 | 139.8s |

##### ImageNet classification:

Below we just evaluate popular CNN architectures that are available (with their trained weights) in `torchvision.models`. Evaluation is done on ImageNet's validation set which consists of 50000 images. In particular we look at accuracy and top-5 accuracy vs. inference time. 

| Model | Accuracy (%) | Top-5 accuracy (%)| Inference time (total) |
| --- | --- | --- | --- |
| AlexNet | 0.57 | 0.79 | 1.6 min |
| VGG16 | 0.72 | 0.90 | 7.8 min |
| Inception v3| 0.78 | 0.94 | 6.4 min |
| ResNet18 | 0.70 | 0.89 | 1.9 min |
| ResNet101 | 0.77 | 0.94 | 7.5 min |