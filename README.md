## TITLE
DEEP LEARNING PROJECT
## PROBLEM STATEMENT
Sign Language Detector
## EXPLANATION
### PREPROCESSING
|:---------------:|:-------------------------------:|
|RGB Dataset|![image](https://github.com/siyagampawar/Sign-Language-Detector/assets/115725393/3bed6f21-69e7-4d14-b217-f193cc9e99d9)|

|GRAYSCALE IMAGING |![image](https://github.com/siyagampawar/Sign-Language-Detector/assets/115725393/d5d5625e-7c87-4683-8746-8ded011ab896)|

|BACKGROUND REMOVAL|![image](https://github.com/siyagampawar/Sign-Language-Detector/assets/115725393/6db68539-e9d7-4f7c-87f7-d61683e951d9)|

|EDGE DETECTION|![image](https://github.com/siyagampawar/Sign-Language-Detector/assets/115725393/566d5c21-70ea-4b9d-b691-4c7dcbdccc4f)|

|     | RGB Dataset                          | Grayscale Imaging                   | Background Removal                  | Edge Detection                      |
|-----|--------------------------------------|-------------------------------------|------------------------------------|-------------------------------------|
|     | ![RGB Dataset](https://github.com/siyagampawar/Sign-Language-Detector/blob/assets/115725393/3bed6f21-69e7-4d14-b217-f193cc9e99d9) | ![Grayscale Imaging](https://github.com/siyagampawar/Sign-Language-Detector/blob/assets/115725393/d5d5625e-7c87-4683-8746-8ded011ab896) | ![Background Removal](https://github.com/siyagampawar/Sign-Language-Detector/blob/assets/115725393/6db68539-e9d7-4f7c-87f7-d61683e951d9) | ![Edge Detection](https://github.com/siyagampawar/Sign-Language-Detector/blob/assets/115725393/566d5c21-70ea-4b9d-b691-4c7dcbdccc4f) |


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


