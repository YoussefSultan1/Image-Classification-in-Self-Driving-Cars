# Image-Classification-in-Self-Driving-Cars
# _**Convolutional Neural Network (CNN) model using Keras**_ ⬇

---


## **Abstract**
Image classification is a fundamental task in the field of computer vision, serving diverse purposes from recognizing objects in images to detecting anomalies in medical scans. This project centers on image classification in the context of self-driving cars, where it holds particular significance. Self-driving vehicles rely on a range of sensor inputs including cameras, lidar, and radar, with high-resolution images playing a crucial role in identifying objects and predicting their behaviors. Notably, accurate classification of vehicles, pedestrians, traffic signs, and road markings contributes substantially to ensuring the safety and reliability of autonomous driving systems.

The project's core objective is the practical implementation of an image classification model using convolutional neural networks (CNNs) to categorize traffic signs – an integral application within the self-driving car domain. The chosen dataset comprises preprocessed RGB images of traffic signs across 43 distinct classes. The primary aim is to train a CNN model capable of effectively classifying traffic sign images. This is realized through the development and training of the CNN model via Keras, followed by an evaluative process employing the accuracy metric. The outcomes of this evaluation offer insights into the efficacy of the adopted approach.

## **Data Set**
The dataset used in this project consisted of preprocessed RGB images of traffic signs belonging to 43 different classes. It included over 50,000 images for training and over 12,000 images for testing. The images were taken from real-world traffic scenarios and were preprocessed to ensure that the signs were centered and normalized in size. This dataset is widely used in the field of computer vision and has been used as a benchmark for evaluating the performance of various image classification models.

![training data](https://github.com/YoussefSultan1/Image-Classification-in-Self-Driving-Cars/assets/99561989/5fac1931-2e0f-48d7-b117-b86fbf3ef842)


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

## **Building Models with Different Filter Sizes**
To explore the impact of different filter sizes on the model's accuracy, a set of models with varying filter sizes was trained. The models were trained for 5 epochs using a batch size of 5 and a learning rate of 0.001. The Learning Rate Scheduler callback function was used to adjust the learning rate during training.

The training and validation accuracy of each model were recorded, and the results are shown below:

![Screenshot 2023-08-20 012430](https://github.com/YoussefSultan1/Image-Classification-in-Self-Driving-Cars/assets/99561989/666ff8de-8212-4cc8-af32-c0d583bccc00)

## **Epoch-Based Model Accuracy Plot**
The plot below compares the accuracy of different models with varying filter dimensions. It shows the training and validation accuracies of each model across epochs. The two subplots display the accuracy of all models with varying filter dimensions, and the legend indicates the filter dimensions used for each model. The results indicate that models with larger filter dimensions achieve higher accuracy in both training and validation, with filter dimensions of 31 and 25 consistently achieving the highest accuracy. Overall, using larger filter dimensions is advisable to achieve higher accuracy, but it is important to balance computational resources with desired accuracy. The accuracy values for each model are also displayed in the code output for detailed comparison.

![accuracy](https://github.com/YoussefSultan1/Image-Classification-in-Self-Driving-Cars/assets/99561989/3e1388e1-678b-461c-b40c-0dd9c13fb80d)

## **Comments and Model Selection**
The process of selecting a model can be a bit subjective and depends on the specific problem, available resources, and personal preferences. However, there are some general guidelines that can help in selecting a model.
In the case of building a set of models with different filter sizes, the aim is to find the best performing model in terms of validation accuracy. The models with larger filter sizes tend to capture more complex features, while models with smaller filter sizes tend to capture simpler features. However, models with larger filters may also be more prone to overfitting, while models with smaller filters may underfit the data.
Looking at the output of the code provided, we can see that the model with filters 3x3 has the highest validation accuracy of 0.88639. This model also has a high training accuracy of 0.98884, indicating that it can capture the complex features in the data without overfitting. On the other hand, models with larger filter sizes such as 31x31 have lower validation accuracy, indicating that they may be overfitting the data.
Therefore, based on the results, it seems reasonable to select the model with filters 3x3 as it has the highest validation accuracy and has a high training accuracy. However, it is important to note that this is not a definitive answer and further experiments, and testing may be needed to determine the best model for the specific problem.

## **Results**
Our model successfully classified the test set example, which indicates that it can accurately predict the class of a given traffic sign image. The training accuracy shows that the model performs well for most of the classes, with only a few classes having lower accuracy, such as "Speed limit (20km/h)" and "Speed limit (30km/h)". We also observed that the model performs better for some classes than others, such as "No vehicles" and "Priority Road", which have high accuracy.

## **Analysis**
The first line of output, (1, 32, 32, 3), indicates the shape of the input image. The image has dimensions of 32x32 pixels and has 3 color channels (RGB).
The first dimension of 1 indicates that there is only one image being inputted.
The second line of output indicates the true label (or class) of the input image. In this case, the true label is class ID 3, which corresponds to the "Speed limit (60km/h)" traffic sign.
The third line of output, (43,), indicates the shape of the output from the model's forward pass. The model predicts the probability scores for each of the 43 possible classes in the dataset.
The fourth line of output classID: 3, indicates the class ID that the model predicted for the input image. In this case, the model predicted that the input image belongs to class 3, which is the correct class.
The fifth line of output, Label: Speed limit (60km/h), indicates the human-readable label corresponding to the predicted class ID. This is the label that we would see on the actual traffic sign in the real world

(1, 32, 32, 3)

[3]

![ex](https://github.com/YoussefSultan1/Image-Classification-in-Self-Driving-Cars/assets/99561989/581bf7f4-8e3c-4e9e-a85e-1ad4f650e50b)

1/1 [==============================] - 0s 44ms/step

(43,)

ClassId: 3

Label: Speed limit (60km/h)

## **Conclusion and future work**
In this project, we built and trained a CNN model using Keras to classify traffic signs. The model achieved a high accuracy on the test set but showed signs of overfitting during training. Further improvements could be made to the model by collecting more datasets to make the car able to classify more objects or applying regularization techniques.
