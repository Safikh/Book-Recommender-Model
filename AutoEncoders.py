# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:37:29 2020

@author: Safiuddin
"""


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import load_npz
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import InputLayer, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD


BATCH_SIZE = 64
EPOCHS = 60
REG = 0.0001


# Loading train and test matrices
train_matrix = load_npz("data/train_sparse.npz")
test_matrix = load_npz("data/test_sparse.npz")

# Creating train and test masks
mask = (train_matrix > 0) * 1.0
mask_test = (test_matrix > 0) * 1.0

# Making copies of data and masks since we will shuffle
train_matrix_copy = train_matrix.copy()
mask_copy = mask.copy()
test_matrix_copy = test_matrix.copy()
mask_test_copy = mask_test.copy()

N, M = train_matrix.shape
print("N: ", N, " M: ", M)
print("N // Batch Size: ", N // BATCH_SIZE)

# Obtaining mean
mu = train_matrix.sum() / mask.sum()
print("mu: ", mu)

# Persisting mean value
with open('data/mu.pkl', 'wb') as f:
  pickle.dump(mu, f)



# Defining Custom loss that applies mask before calculating loss
def custom_loss(y_true, y_pred):
    '''
    Custom loss function that MSE after applying mask
    Inputs : 
             y_true: train data batch
             y_pred: output of AutoEncoder model
             
    Returns Mean squared error for non-empty ratings

    '''
    mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
    diff = y_pred - y_true
    sqdiff = diff * diff * mask
    sse = K.sum(K.sum(sqdiff))
    n = K.sum(K.sum(mask))
    return sse / n

def generator(train_matrix, M):
    '''
    Generator that yields train data after applying masks and scaling the data
    Inputs : 
             train_matrix: Sparse Matrix of shape NxM containing train data
             M           : Sparse Matrix of shape NxM containing mask for train data
             
    Returns: 
             noisy       : Numpy array of train data batch (X)
             train       : Numpy array of train data batch (y)
    '''

    while True:
        train_matrix, M = shuffle(train_matrix, M)
        for i in range(train_matrix.shape[0] // BATCH_SIZE + 1):
            upper = min((i+1)*BATCH_SIZE, train_matrix.shape[0])
            train = train_matrix[i*BATCH_SIZE:upper].toarray()
            m = M[i*BATCH_SIZE:upper].toarray()
            train = train - mu * m # Must keep zeros at zero
            noisy = train # Noise added through Dropout
            yield noisy, train
            
def test_generator(train_matrix, M, test_matrix, M_test):
    '''
    Generator that yields train and test data after applying masks and scaling the data
    Inputs : 
             train_matrix: Sparse Matrix of shape NxM containing train data
             M           : Sparse Matrix of shape NxM containing mask for train data
             test_matrix : Sparse Matrix of shape NxM containing test data
             M_test      : Sparse Matrix of shape NxM containing mask for test data
    Returns: 
             train : Numpy array of train data batch
             test  : Numpy array of test data batch 
    '''
    
    while True:
        for i in range(train_matrix.shape[0] // BATCH_SIZE + 1):
            upper = min((i+1)*BATCH_SIZE, train_matrix.shape[0])
            train = train_matrix[i*BATCH_SIZE:upper].toarray()
            m = M[i*BATCH_SIZE:upper].toarray()
            test = test_matrix[i*BATCH_SIZE:upper].toarray()
            mt = M_test[i*BATCH_SIZE:upper].toarray()
            train = train - mu * m
            test = test - mu * mt
            yield train, test


# Building the model
# Layers, optimizer and Hyperparameters were selected after tuning several times
model = Sequential()
model.add(InputLayer(input_shape=(M,)))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu', kernel_regularizer=l2(REG)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu', kernel_regularizer=l2(REG)))
model.add(Dropout(0.5))
model.add(Dense(30, activation='tanh', kernel_regularizer=l2(REG)))
model.add(Dense(M, kernel_regularizer=l2(REG)))


model.compile(
    loss=custom_loss,
    optimizer=SGD(lr=0.08, momentum=0.9),
    metrics=[custom_loss]
    )

history = model.fit(
    generator(train_matrix, mask),
    validation_data=test_generator(train_matrix_copy, mask_copy, test_matrix_copy, mask_test_copy),
    epochs=EPOCHS,
    steps_per_epoch=train_matrix.shape[0] // BATCH_SIZE + 1,
    validation_steps=test_matrix.shape[0] // BATCH_SIZE + 1)


# plot losses
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(history.history['custom_loss'], label="train mse")
plt.plot(history.history['val_custom_loss'], label="test mse")
plt.legend()
plt.show()

# Persisting the model
model.save('data/AE_model.h5')
model2 = tf.keras.models.load_model('data/AE_model.h5', compile=False) 

