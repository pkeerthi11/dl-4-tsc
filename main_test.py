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

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)

# change this directory for your machine
root_dir = '/home/pkeerthi/projects/def-sblain/pkeerthi/BiomusicDeepLearning/'
output_directory = '/home/pkeerthi/projects/def-sblain/pkeerthi/DeepLearningTrainingResults/'

archive_name = sys.argv[1]
dataset_name = sys.argv[2]
classifier_name = sys.argv[3]
itr = sys.argv[4]

datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

x_train = datasets_dict[dataset_name][0]
y_train = datasets_dict[dataset_name][1]
x_test = datasets_dict[dataset_name][2]
y_test = datasets_dict[dataset_name][3]

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

# transform the labels from integers to one hot vectors
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(y_test, axis=0).reshape(-1, 1)
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(x_test.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension 
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

input_shape = x_test.shape[1:]
classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory + classifier_name, verbose = True)

model_path = output_directory + classifier_name + '/best_model.hdf5'
model = keras.models.load_model(model_path)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)



print("Accuracy: ", acc, " Balanced accuracy: ", bal_acc)
