# Seq2Seq Machine Translation Model

This project demonstrates a sequence-to-sequence (seq2seq) model built for machine translation, specifically translating English sentences to Spanish. The model is implemented using Keras, leveraging an LSTM (Long Short-Term Memory) architecture for both the encoder and decoder components.

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Inference](#inference)
7. [MLflow Integration](#mlflow-integration)
8. [Conclusion](#conclusion)

## Introduction

This project aims to build a seq2seq model to translate sentences from English to Spanish. The model utilizes Keras with LSTM layers to create both the encoder and decoder for sequence translation. Additionally, the model is integrated with MLflow for tracking and managing experiments.

## Requirements

Before running the code, make sure you have the following dependencies installed:

#Data Preprocessing
Data Preprocessing
Loading Data: The dataset is read from a CSV file using pandas. The dataset is limited to the first 1500 rows for training purposes.
Tokenization: The English and Spanish sentences are tokenized using the Keras Tokenizer class. Each word in the sentences is assigned an integer index.
Padding: The sequences are padded using pad_sequences to ensure that all input and output sequences have the same length.
Target Expansion: The output sequences (Spanish sentences) are expanded to match the shape expected by the model, adding an extra dimension to the target sequences.

#Model Architecture
The model consists of two main components:

Encoder
The encoder takes the English sentence as input and converts it into a fixed-size context vector (state vectors state_h and state_c) using an LSTM layer.
An embedding layer is used to map the input tokens into dense vectors.
Decoder
The decoder takes the context vector from the encoder and generates the translated Spanish sentence word by word.
Like the encoder, the decoder has an LSTM layer and an embedding layer. The decoder also uses a dense layer with a softmax activation to output probabilities for each word in the Spanish vocabulary.

#Training
The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss. The model is trained for 35 epochs with a batch size of 8.

#Inference
To translate a new sentence, the model uses two separate parts:

Encoder Model: Converts the input sentence into a fixed-size context vector.
Decoder Model: Generates the translated sentence, predicting one word at a time.

#MLflow Integration
MLflow is integrated into the project to track experiments, log model parameters, and save the trained model. This allows for versioning and easy deployment.

#Conclusion
This project demonstrates how to build a seq2seq machine translation model using LSTM networks in Keras, along with data preprocessing, training, and inference. MLflow is used to track experiments and save the trained models.

By following this approach, you can extend the model to other language pairs and further improve its accuracy and performance





```bash
pip install pandas numpy tensorflow mlflow


