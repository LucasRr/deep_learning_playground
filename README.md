This repo is a playground to implement, train, and benchmark various deep learning models on a variety of tasks. All models are implemented in Pytorch. 

Most experiments can be found as a collection of notebooks (with comments).


##### Example notebooks:

- `MNIST_classification_MLP.ipynb`: test various multi-layer perceptron architectures for MNIST classification
- `MNIST_classification_CNN.ipynb`: test various CNN architectures for MNIST classification
- `CIFAR10_classification.ipynb`: implement and train (from scratch) various popular CNN-based architectures (LeNet5, AlexNet, VGG, Inception, ResNet) for CIFAR10 classification 
- `CNN_for_image_denoising.ipynb`: compare filtering with various fixed or learned convolution kernel, and simple mutlilayer CNNs for image denoising. Evaluated on MNIST
- `ImageNet_classification_pretrained.ipynb`: benchmark various pretrained models (available from Pytorch's `torchvision`) for ImageNet classification, in terms of accuracy vs. inference speed.
- `transfer_learning.ipynb`: use transfer learning to classify CIFAR10 images using models pre-trained on MNIST


Below are some benchmark results:

##### CIFAR-10 classification:

The networks below were adapted from popular image classifiers which achieved high performance on ImageNet, and trained from scratch. In particular the networks were adapted to take 32x32 images as input, with a relatively low number of layers (compared to the original implementations) in order to avoid overfitting and keep training time reasonable. For these reasons the networks are simplified version of the original proposed networks and the results might not reflect results demonstrated on ImageNet.

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
| ResNet1012 | 0.77 | 0.94 | 7.5 min |