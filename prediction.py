# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:45:29 2020

@author: Safiuddin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import save_npz, load_npz
from scipy.sparse import lil_matrix

import tensorflow as tf

movies = pd.read_csv('data/processed_movies')
model = tf.keras.models.load_model('AE_model.h5', compile=False)

A = load_npz("data/train_sparse.npz")
A_test = load_npz("data/train_sparse.npz")
mask = (A > 0) * 1.0
mu = A.sum() / mask.sum()

n = np.random.randint(699)
pred = model.predict(A_test[n].toarray() - mu) + mu
print(n)
print(pred[pred < 0].shape)
print(pred[pred > 0].shape)
print(pred[pred > 1].shape)
print(pred[pred > 2].shape)
print(pred[pred > 3].shape)
print(pred.mean())

pred_dict = dict(enumerate(pred[0]))
prediction_df = movies.copy()
prediction_df['predicted_rating'] = prediction_df['new_id'].map(pred_dict)
test_rating_dict = dict(enumerate(A_test[n].toarray()[0]))
prediction_df['test_rating'] = prediction_df['new_id'].map(test_rating_dict)
train_rating_dict = dict(enumerate(A[n].toarray()[0]))
prediction_df['train_rating'] = prediction_df['new_id'].map(train_rating_dict)

rand_list = np.random.randint(499, size=(20,))
movies[movies['new_id'].isin(rand_list)][['new_id', 'name']]
