# Image-Classification-in-Self-Driving-Cars
# _**Here's a brief overview of what code is doing:**_ ⬇

---


## **Abstract**
Image classification is a fundamental task in the field of computer vision, serving diverse purposes from recognizing objects in images to detecting anomalies in medical scans. This project centers on image classification in the context of self-driving cars, where it holds particular significance. Self-driving vehicles rely on a range of sensor inputs including cameras, lidar, and radar, with high-resolution images playing a crucial role in identifying objects and predicting their behaviors. Notably, accurate classification of vehicles, pedestrians, traffic signs, and road markings contributes substantially to ensuring the safety and reliability of autonomous driving systems.

The project's core objective is the practical implementation of an image classification model using convolutional neural networks (CNNs) to categorize traffic signs – an integral application within the self-driving car domain. The chosen dataset comprises preprocessed RGB images of traffic signs across 43 distinct classes. The primary aim is to train a CNN model capable of effectively classifying traffic sign images. This is realized through the development and training of the CNN model via Keras, followed by an evaluative process employing the accuracy metric. The outcomes of this evaluation offer insights into the efficacy of the adopted approach.

Encompassing over 50,000 training images and exceeding 12,000 test images, the dataset authentically reflects real-world traffic scenarios. Preprocessing procedures are employed to standardize the images, centering the traffic signs and normalizing sizes. This dataset stands as an esteemed benchmark in the realm of computer vision, widely recognized for benchmarking the performance of various image classification models. Overall, this project presents a comprehensive exploration into image classification, shedding light on the deployment of CNNs for discerning traffic signs – a pivotal facet within the self-driving car landscape.

## **Data Preprocessing**
Before using the dataset in this project, some preprocessing steps were performed.
The data was loaded from a binary pickle file and stored in a dictionary named "data", containing keys for training, testing, and validation sets. The input images had dimensions of 32 by 32 pixels with three color channel (RGB), the images were transposed. Additionally, the labels were converted to categorical variables using the to_categorical function from the Keras library. These preprocessing steps helped ensure that the data was formatted correctly for use in training the neural network model.

## **Main Model Architecture and Configuration**
• **Convolutional layer**: The model uses a convolutional layer with 32 filters of size 3x3. This layer applies the filters to the input image to extract features from it.

• **Max-pooling layer**: A max-pooling layer of size 2 follows the convolutional layer. This layer reduces the dimensionality of the features while preserving the most essential information.

• **Flattening layer**: The output of the max-pooling layer is flattened to create a one-dimensional vector that can be used as input to a dense layer.

• **Dense layer**: The model has a dense layer with 500 neurons and a Relu activation function. This layer takes the flattened vector as input and applies a set of weights to it to produce an intermediate representation of the input.
Output layer: The final dense layer has 43 neurons and a SoftMax activation function. This layer produces the probability distribution over the 43 classes of traffic signs.

• **Code Output**: some examples of the training data.

• **Loss function**: The model was compiled with categorical cross-entropy loss function. This is a common loss function used in classification tasks that penalizes the model for making incorrect predictions.

• **Optimizer**: The Adam optimizer was used to minimize the loss of function. Adam is an adaptive learning rate optimization algorithm that is widely used in deep learning.

Overall, this architecture and configuration were chosen based on previous work in traffic sign classification and proved to be effective in achieving high accuracy on this dataset.
## **Model Training with Limited Data**
To evaluate the model's performance with a small amount of data, we trained it for 15 epochs with a batch size of 32 and a learning rate of 0.001. While this resulted in overfitting due to the limited dataset, it allowed us to evaluate the model using the accuracy metric and gain insights into its potential performance with more data. During training, we used the Learning Rate Scheduler callback function to adjust the learning rate. This can be seen from the accuracy and loss plots shown below:

![overfitting small data](https://github.com/YoussefSultan1/Image-Classification-in-Self-Driving-Cars/assets/99561989/f3d21658-39ee-44f9-af3a-be3f7b97c775)


## **Hyperparameters Tuning**
Hyperparameters are the configuration settings of a machine learning algorithm that cannot be learned from data during the training process. They must be set prior to training and can significantly impact the performance of the resulting model.
In our code, hyperparameter tuning is achieved by exploring the effect of different filter sizes in convolutional layers. A set of models are trained using different filter sizes, and the performance of each model is evaluated to determine the optimal filter size. This process allows us to find the best configuration of hyperparameters that results in the highest model accuracy. The following hyperparameters were used for training the model:
• Learning rate: 1e-3
• Learning rate decay: 0.95
• Number of epochs: 15
• Optimizer: Adam
• Loss function: Categorical cross-entropy
