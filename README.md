## TITLE
DEEP LEARNING PROJECT
## PROBLEM STATEMENT
Sign Language Detector
## EXPLANATION
### PREPROCESSING
|                |                                                                                                     |
|----------------|-----------------------------------------------------------------------------------------------------|
| RGB Dataset    |![image](https://github.com/siyagampawar/Sign-Language-Detector/assets/115725393/d8cd9d77-c7a6-4080-9ad5-04341e0a90e2)|
| Grayscale Imaging |![image](https://github.com/siyagampawar/Sign-Language-Detector/assets/115725393/79f0fedb-d4dc-497d-8d55-167c39f2d550)|
| Background Removal |![image](https://github.com/siyagampawar/Sign-Language-Detector/assets/115725393/148561c2-fc2e-417f-9f18-9e9e82de2959)|
| Edge Detection | ![image](https://github.com/siyagampawar/Sign-Language-Detector/assets/115725393/898a5c55-6156-4e61-8338-28d3ce3d71f9)|

### MODELS

1.CNN
Convolutional Neural Networks (CNNs) are deep learning models designed for image processing tasks, inspired by the visual cortex's organization. CNNs utilize convolutional layers to extract features from input images, pooling layers to reduce dimensionality, and fully connected layers for classification. They excel in tasks like image classification, object detection, and semantic segmentation. Popular architectures include LeNet-5, AlexNet, VGGNet, ResNet, and MobileNet. With their ability to learn hierarchical representations, CNNs have revolutionized computer vision, powering applications ranging from autonomous vehicles to medical diagnosis, with pre-trained models and frameworks like TensorFlow and PyTorch facilitating implementation.
ACCURACY : 100 %

2.VGG 16
VGG16 is a deep convolutional neural network architecture known for its simplicity and effectiveness. Developed by the Visual Geometry Group at the University of Oxford, it comprises 16 weight layers, including 13 convolutional layers and 3 fully connected layers. VGG16 features small 3x3 convolutional filters with a stride of 1 and max-pooling layers to downsample feature maps. Despite its depth, VGG16 maintains a uniform architecture with no complex elements like inception modules or residual connections. It achieved notable success in image classification tasks, particularly winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2014, showcasing its capability and robustness in large-scale image recognition.

3.RESNET
ResNet, short for Residual Network, is a deep convolutional neural network architecture renowned for addressing the vanishing gradient problem in training very deep networks. Introduced by Kaiming He et al., ResNet introduces skip connections or residual connections, enabling the flow of gradients through the network, even in extremely deep architectures. These connections allow the network to learn residual functions, effectively learning the difference between the input and output of each layer. As a result, ResNet can effectively train networks with hundreds or even thousands of layers, achieving state-of-the-art performance in various computer vision tasks such as image classification, object detection, and image segmentation.
ACCURACY : 100 %

4.EFFICIENTNET
EfficientNet is a convolutional neural network architecture designed to achieve state-of-the-art performance while maintaining efficiency in terms of model size and computational resources. Developed by Mingxing Tan and Quoc V. Le, EfficientNet scales the network's depth, width, and resolution simultaneously using a compound scaling method. By efficiently balancing these dimensions, EfficientNet achieves superior performance compared to traditional architectures. It introduces compound scaling to optimize model parameters, enabling better utilization of computational resources. EfficientNet has demonstrated remarkable success across various computer vision tasks, including image classification, object detection, and segmentation, making it a preferred choice for applications requiring high performance in resource-constrained environments.
ACCURACY : 99.87%


## DATASET LINK
https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl


