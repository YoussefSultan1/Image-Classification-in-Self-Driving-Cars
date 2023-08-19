# Image-Classification-in-Self-Driving-Cars
# _**Here's a brief overview of what code is doing:**_ ⬇

---


## **Data Acquisition**
The script begins with connecting to Google Drive to load the data and uploading any necessary files. Then, it downloads a dataset from Kaggle, which appears to be a preprocessed dataset of traffic signs.

## **Data Preparation**
The downloaded dataset is a pickle file, so it's loaded into memory. The labels are converted to one-hot encoded format, and the data is reshaped so that the channels are last. Also, it only keeps 60% of the training data to avoid overloading RAM.

## **Exploratory Data Analysis**
The script visualizes some training examples and saves the plot.

## **Model Definition**
The script defines a Convolutional Neural Network (CNN) model using Keras. The model consists of a convolutional layer, a max pooling layer, a flatten layer, a dense layer, and an output layer with a softmax activation function.

## **Model Compilation and Training**
The model is compiled using the Adam optimizer and the categorical cross-entropy loss function. Then, it's trained on a small subset of data for demonstrating overfitting. It uses a learning rate scheduler callback during the training process.

## **Performance Evaluation**
The script plots the training and validation accuracy of the model and provides a summary of the training results. Then it saves these plots for future use.

## **Hyperparameter Tuning**
The script trains multiple models with different filter sizes for the convolutional layer, saves these models, and compares their performance.

## **Final Evaluation**
The script reloads the saved models and their training histories and compares their performance using plots. It also performs predictions on the test dataset and evaluates their accuracy.

## **Prediction on Single Image**
The script selects an individual test image and predicts its class using the trained models.
