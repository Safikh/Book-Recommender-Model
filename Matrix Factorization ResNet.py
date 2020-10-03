# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:38:09 2020

@author: Safiuddin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dot, Add
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam

df = pd.read_csv('data/processed_ratings.csv')
df = df.dropna() # To remove any unintended nan values
N = df['user_id'].nunique()
M = df['movie_id'].nunique()

df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# Initialize parameters
K = 25 # Latent dimensionality
mu = df_train['rating'].mean()
EPOCHS = 25
reg = 0.0005 # Regularization penalty


# Keras Model Architecture
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K)(u) # (N, 1, K)
m_embedding = Embedding(M, K)(m) # (N, 1, K)

# Main branch
u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u) # (N, 1, 1)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m) # (N, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding]) # (N, 1, 1)

x = Add()([x, u_bias, m_bias])
x= Flatten()(x) # (N, 1)


# Side Branch
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
y = Concatenate()([u_embedding, m_embedding]) # (N, 2K)
y = Dense(50, kernel_regularizer=l2(reg))(y)
'''
y = BatchNormalization()(y)
y = Activation('elu')(y)
y = Dropout(0.6)(y)
y = Dense(100)(y)
'''
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Dropout(0.6)(y)
y = Dense(1)(y)

# Merge
x = Add()([x, y])

model = Model(inputs=[u,m], outputs=x)
model.compile(
    loss='mse',
    # optimizer='Adam',
    optimizer=Adam(lr=0.001),
    # optimizer=SGD(lr=0.01, momentum=0.9),
    metrics=['mse'],
    )

r = model.fit(
    x=[df_train['user_id'].values, df_train['movie_id'].values],
    y=df_train['rating'].values - mu,
    epochs=EPOCHS,
    batch_size=128,
    validation_data=(
        [df_test['user_id'].values, df_test['movie_id'].values],
        df_test['rating'].values - mu
        )
    )

# plot losses
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='test loss')
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mse'], label='train mse')
plt.plot(r.history['val_mse'], label='test mse')
plt.legend()
plt.show()

model.save('data/MFRes.h5')

model2 = tf.keras.models.load_model('data/MFRes.h5')
preds = model2.predict([df_test['user_id'].values, df_test['movie_id'].values]) + mu


preds = preds.clip(0.5, 5)
preds = np.round(preds, 1)

fig = plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
sns.countplot(df_test['rating'], color = 'r', saturation = 1)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution of Actual Ratings')
plt.subplot(2, 1, 2)
sns.countplot(preds.flatten(),color = 'b', saturation = 1)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution of Predicted Ratings')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=0.5)
plt.show()