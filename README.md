# PCB-component-recognition

## Electronic Components Classification
This project aims to classify various electronic components (ECs) using a deep learning model based on the InceptionV3 architecture. The model is trained to identify different types of electronic components from images. The database with labeled images of various ECs on PCBs can be found [here](https://universe.roboflow.com/roboflow-100/printed-circuit-board).


## Introduction
The goal of this project is to develop a deep learning model that can accurately classify electronic components based on PCBs from images. The model is built using TensorFlow and Keras, and it leverages the InceptionV3 architecture pre-trained on ImageNet.

## Features
Classification of various electronic components: The model is designed to classify different types of electronic components such as buttons, capacitors, inductors, resistors, etc.
Use of transfer learning with InceptionV3: The model utilizes the InceptionV3 architecture pre-trained on ImageNet, which helps in achieving better performance with less training data.
Customizable hyperparameters: The model allows for customization of hyperparameters such as learning rate, batch size, and number of epochs.
Evaluation metrics: The model provides evaluation metrics including confusion matrix and classification report to assess its performance.

## Neural Network Structure
The neural network is built using the InceptionV3 architecture as the base model, followed by additional layers for classification. Here is a detailed breakdown of the network structure:

### Base Model (InceptionV3):

The InceptionV3 model is used as the base, pre-trained on ImageNet.
The input shape is (299, 299, 3), which is the expected input size for InceptionV3.
The top layers of InceptionV3 are excluded (include_top=False).
All layers of the base model are set to non-trainable to leverage the pre-trained weights.

### Global Average Pooling Layer:

A GlobalAveragePooling2D layer is added on top of the base model to reduce the spatial dimensions of the output.
Dense Layer:
A dense layer with 128 units and ReLU activation function is added.
L2 regularization with a factor of 0.00066172 is applied to the kernel to prevent overfitting.
Dropout Layer:
A dropout layer with a rate of 0.3 is added to further prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
Output Layer:
A dense layer with 1 unit and sigmoid activation function is added to produce the final output. This layer outputs a probability score for the binary classification task.
Model Compilation
The model is compiled using the Adam optimizer with a learning rate of 0.001. The loss function used is binary cross-entropy, and the evaluation metric is accuracy.

### Training
The model is trained for 20 epochs with a batch size of 32. Class weights are computed to handle class imbalance and are used during training. The training process includes validation on a separate validation set to monitor the model's performance.

### Evaluation
The model is evaluated on a test set, and the following metrics are reported:

Test Accuracy: The accuracy of the model on the test set.
Confusion Matrix: A matrix used to evaluate the accuracy of a classification.
Classification Report: A report that includes precision, recall, F1-score, and support for each class.

## Code Overview
The code includes the following steps:

Loading Images and Labels: Images and their corresponding labels are loaded from specified directories.
Normalization: Images are normalized by dividing by 255.0.
Class Weights Calculation: Class weights are computed to handle class imbalance.
Model Definition: The neural network model is defined using the InceptionV3 base model and additional layers.
Model Compilation: The model is compiled with the specified optimizer, loss function, and metrics.
Model Training: The model is trained on the training set with validation on the validation set.
Model Evaluation: The model is evaluated on the test set, and evaluation metrics are reported.

## Results

Accuracy of classification is dependent on the nature of the EC considered. It remains relatively accurate for  
