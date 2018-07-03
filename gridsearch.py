# Dr David Ellison - June 15, 2018
# The purpose of this code is to do a grid search on various hyperparameters of a simple neural network. 


# code at: /home/dellison/code/kr_hyperparameter/kr_it_GridSearchCV.py
# singularity shell -B /home/dellison --nv tensorflow-18.04-py3_de.simg
# installed keras scikit-learn


# Load libraries
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
import time

# Set random seed
np.random.seed(0)

# Number of features
number_of_features = 100

# Generate features matrix and target vector
features, target = make_classification(n_samples = 10000, n_features = number_of_features, n_informative = 3, n_redundant = 0, n_classes = 2, weights = [.5, .5], random_state = 0)

# Create function returning a compiled network
def create_network(optimizer='rmsprop'):
    # Start neural network
    network = models.Sequential()
    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))
    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu'))
    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))
    # Compile neural network
    network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) # Accuracy performance metric
    # Return compiled network
    return network

# Wrap Keras model so it can be used by scikit-learn
neural_network = KerasClassifier(build_fn=create_network, verbose=0)

# Create hyperparameter space
epochs = [2, 4]
batches = [1, 5, 10]
optimizers = ['rmsprop', 'adam']

# Create hyperparameter options
hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)
#hyperparameters = dict(epochs=epochs, batch_size=batches)

# Create grid search
grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters)

# Fit grid search
t1 = time.time ()
grid_result = grid.fit(features, target)
t2 = time.time ()
run_time_in_min = round(((t2-t1)/60),2) 
# View hyperparameters of best neural network
#grid_result.best_params_
print("Total time for GridSearchCV:")
print("   ",run_time_in_min," minutes")
print("Total parameters searched:")
print("   ",hyperparameters)
print("Best parameters are:")
print("   ",grid_result.best_params_)
print("With a score of:")
print("   ",grid_result.best_score_)

