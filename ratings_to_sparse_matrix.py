# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:14:41 2020

@author: Safiuddin
"""

# Importing libraries
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, save_npz


# Loading data
df = pd.read_csv('data/processed_ratings.csv')


N = df['user_id_new'].nunique() # Number of users
M = df['book_id_new'].nunique() # Number of books

print("N: {}, M: {}".format(N, M))

# Train-Test-Split
df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]


# Creating a lil matrix of shape (N, M) for train data
train_mat = lil_matrix((N, M))

def convert_train(row):
    
    i = int(row['user_id_new']) - 1
    j = int(row['book_id_new']) - 1
    train_mat[i, j] = row['rating']

df_train.apply(convert_train, axis=1)


# Creating a lil matrix of shape (N, M) for test data
test_mat = lil_matrix((N, M))

def convert_test(row):
    
    i = int(row['user_id_new']) - 1
    j = int(row['book_id_new']) - 1
    test_mat[i, j] = row['rating']

df_test.apply(convert_test, axis=1)

# Converting from lil matrix to csr matrix
train_mat = train_mat.tocsr()
test_mat = test_mat.tocsr()

# Saving the csr matrices
save_npz('data/train_sparse.npz', train_mat)
save_npz('data/test_sparse.npz', test_mat)
