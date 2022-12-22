This repo is a playground to experiment with various deep learning models, on a variety of tasks, such as image classification, image denoising, dimensionality reduction, transfer learning etc. All models are implemented in Pytorch. 

Most experiments can be found as a collection of notebooks (see the list below).

This repo is work in progress and more notebooks/experiments will be added in the future. 

#### Requirements

Most models have been trained (when possible) on Apple M1 GPU, which is currently possible using pytorch-nightly, which can be installed with:

```
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

Change `device = "mps"` to `device = "cpu"` or `"cuda"` in the code to train/evaluate on CPU or cuda GPU instead. 

See `requirements.txt` for more details on the environment. 

#### List of notebooks:

- `MNIST_classification_MLP.ipynb`: multi-layer perceptron architectures for MNIST classification
- `MNIST_classification_CNN.ipynb`: CNN architectures for MNIST classification
- `CIFAR10_classification.ipynb`: popular CNN-based architectures (LeNet5, AlexNet, VGG, Inception, ResNet) for CIFAR10 classification (implemented and trained from scratch)
- `CNN_for_image_denoising.ipynb`: compare filtering with various fixed or learned convolution kernels, and simple mutlilayer CNNs for image denoising. Evaluated on MNIST
- `ImageNet_classification_pretrained.ipynb`: benchmark various pretrained models (available from Pytorch's `torchvision`) for ImageNet classification, in terms of accuracy vs. inference speed.
- `transfer_learning.ipynb`: transfer learning to classify CIFAR10 images using models pre-trained on ImageNet
- `network_visualization_activation_maximization.ipynb`: visualize convolution filters of popular CNN models, perform activation maximization for network visualization, and experiment with adversarial examples. 
- `autoencoders_MNIST.ipynb`: compare linear, fully-connected and convolutional autoencoders on MNIST
- `autoencoders_face_images.ipynb`: experiment with autoencoders for face representation, generation and reconstruction


---


Below are some benchmark results:

#### CIFAR-10 classification:

A comparison of popular architectures on the CIFAR-10 dataset. Note that those are my own implementations, which have been adapted (and simplified) to take 32x32 images as input, and trained from scratch. The results might thus not be optimal and might not reflect the results demonstrated on ImageNet.

| Model | Test accuracy (%) | Training time (per epoch)|
| --- | --- | --- |
| Fully-connected | 0.50 | 5.3s|
| LeNet/AlexNet-style network | 0.66 | 5.9s |
| VGG-style network | 0.71 | 10.1s|
| Inception-style network | 0.76 | 15.6s |
| ResNet-style network | 0.66 | 12.1s |
| Transfer learning (ImageNet-AlexNet features) | 0.83 | 92.5s |
| Transfer learning (ImageNet-ResNet18 features) | 0.78 | 139.8s |

#### ImageNet classification:

Below we benchmark popular (pre-trained) CNN architectures that are available (with their trained weights) in `torchvision.models` or timm. Evaluation is done on ImageNet's validation set which consists of 50000 images. In particular we look at accuracy and top-5 accuracy vs. inference time. 

| Model | Accuracy (%) | Top-5 accuracy (%)| Inference time (total) |
| --- | --- | --- | --- |
| AlexNet | 0.57 | 0.79 | 1.6 min |
| VGG16 | 0.72 | 0.90 | 7.8 min |
| Inception v3| 0.78 | 0.94 | 6.4 min |
| ResNet18 | 0.70 | 0.89 | 1.9 min |
| ResNet101 | 0.77 | 0.94 | 7.5 min |
| EfficientNet_b2 | 0.78 | 0.94 | 7.4 min |