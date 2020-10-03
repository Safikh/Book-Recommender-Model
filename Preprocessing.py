# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:27:20 2020

@author: Safiuddin
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# Loading data
books = pd.read_csv('data/books.csv')
ratings = pd.read_csv('data/ratings.csv')


# Taking top 9500 books based on number of ratings
top_9500_books = ratings['book_id'].value_counts()[:9500].index 
ratings = ratings[ratings['book_id'].isin(top_9500_books)] 


# Taking top 15000 users based on number of ratings
top_15000_users = ratings['user_id'].value_counts()[:15000].index
ratings = ratings[ratings['user_id'].isin(top_15000_users)]

ratings = ratings.rename(columns={'user_id': 'user_id_old', 'book_id': 'book_id_old'})


# Creating a list of user_ids from 0 to 14999
user_id_map = dict(zip(ratings['user_id_old'].unique(), range(ratings['user_id_old'].nunique())))
ratings['user_id_new'] = ratings['user_id_old'].map(user_id_map)


# Creating a list of book_ids from 0 to 9499
book_id_map = dict(zip(ratings['book_id_old'].unique(), range(ratings['book_id_old'].nunique())))
ratings['book_id_new'] = ratings['book_id_old'].map(book_id_map)


# Adding title from books dataset to ratings dataset using LEFT Join
df = ratings.merge(books[['id', 'title']], how='left', left_on='book_id_old', right_on='id', suffixes=[None, '_old_2'])
df = df.drop(columns=['id']) # Dropping duplicate ID column from books dataset
df.to_csv('data/processed_ratings.csv', index=None)


books['book_id_new'] = books['id'].map(book_id_map) # Adding new book_id to books dataset
books = books[['best_book_id', 'book_id_new', 'title', 'small_image_url', 'original_publication_year', 'average_rating', 'authors']] # Keeping the relevant columns
books = books[books['book_id_new'].notnull()] # Keeping the 9500 top books


books['medium_image_url'] = books['small_image_url'].apply(lambda url: re.sub(r"(\d)[s]", r"\1m", url)) # Replacing 's' with 'm' in url to access medium sized images
books['large_image_url'] = books['small_image_url'].apply(lambda url: re.sub(r"(\d)[s]", r"\1l", url)) # Replacing 's' with 'l' in url to access medium sized images
books.to_csv('data/processed_books.csv', index=None)
