# Facial Expression Recognition using Deep Learning

This repository contains a project that uses deep learning techniques to recognize facial expressions from images. The project utilizes the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) from Kaggle, which consists of images labeled with one of seven emotional categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Project Overview

The goal of this project is to develop a deep learning model that can predict the emotion conveyed by a human face in an image. The project employs Convolutional Neural Networks (CNN) to classify the facial expressions based on image pixel data.

## Files in this Repository

- **face_expression_recognition_colab.ipynb**: Colab notebook containing the code to train and test the facial expression recognition model.
- **model.h5**: Pre-trained model for facial expression classification (optional, if available).
- **README.md**: This file with the project details.

## Requirements

To run this project, you need to install the following Python libraries:

- TensorFlow (2.x)
- Keras
- NumPy
- Matplotlib
- Pandas
- OpenCV
- scikit-learn

## Running the Project

1. Clone the repository or download the dataset from [Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).
2. Open the provided Google Colab notebook (`face_expression_recognition_colab.ipynb`).
3. Follow the instructions inside the notebook to load the dataset, preprocess the images, build the model, and train the neural network.
4. Evaluate the model on a test set of images and analyze the results.
