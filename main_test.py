# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:37:15 2023

@author: pkeer
"""

from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import tensorflow.keras as keras
import tensorflow as tf
import os
import numpy as np
import h5py
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# change this directory for your machine
root_dir = '/home/pkeerthi/projects/def-sblain/pkeerthi/BiomusicDeepLearning/'
output_directory = '/home/pkeerthi/projects/def-sblain/pkeerthi/DeepLearningTrainingResults/'

archive_name = sys.argv[1]
dataset_name = sys.argv[2]
classifier_name = sys.argv[3]
itr = sys.argv[4]

datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
x_test = datasets_dict[dataset_name][2]
y_test = datasets_dict[dataset_name][3]

nb_classes = len(np.unique(y_test, axis=0))

# transform the labels from integers to one hot vectors
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(y_test, axis=0).reshape(-1, 1)
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(x_test.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension 
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

input_shape = x_test.shape[1:]


model_path = output_directory + classifier_name + '/best_model.hdf5'
model = keras.models.load_model(model_path)
y_pred = model.predict(x_test)

acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)

print("Accuracy: ", acc, "Balanced accuracy: ", bal_acc)