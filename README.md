# Book-Recommender-Model

Recommender System for books using AutoEncoders

Dataset: The Dataset used for ratings and books information is from goodbooks-10k. It is available at https://www.kaggle.com/zygmunt/goodbooks-10k . Ratings dataset has been converted into sparse matrix grid. Cover photos were taken from the URLs provided with the dataset. The Dataset is reusable under CC BY-SA 4.0 License.

Methodology: Ratings data was converted into a grid of shape NxM using Scipy sparse-matrix to reduce storage. The data was split into train and test matrices. An Auto-Encoder model was built with three layers of nodes 30, 10 and 30 respectively. Dropout layers and L2 Regularization were added to reduce overfitting. Custom-loss function was created that only calculated the loss for positive ratings.

Hyper-Parameter tuning was performed to achieve minima. Goal of the training was to achieve a reasonable minima without using too many layers or nodes, as that would result in duplicating the input layer to create the output layer. Final output recommendations are randomized so that every click would result in a new set of recommendations.Flask was used for creating web app and Heroku for deployment.

Deployment Code can be found in my other repository: https://github.com/Safikh/Book-Recommender
