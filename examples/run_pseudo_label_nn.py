#!/usr/bin/env python
# encoding=utf-8

from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

class DNN(object):
	"""
	Define a DNN model for classification.
	"""

	def __init__(self, batch_size=128):
		self.batch_size = batch_size

	def build_model(self, input_dim, output_dim, hidden_dim_list=[128, 50]):
		'''
		:param inputdim: int type, the dim of input data.
		:param outputdim: int type, the number of class.
		'''
		model = Sequential()
		model.add(Dense(hidden_dim_list[0], input_dim=input_dim, activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		for i in range(1, len(hidden_dim_list)):
			model.add(Dense(hidden_dim_list[i], activation='relu'))
			model.add(BatchNormalization())
			model.add(Dropout(0.5))
		model.add(Dense(output_dim, activation='softmax'))

		return model

	

# normalization
def normalize(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))

def get_data():
	X, y = datasets.load_breast_cancer(return_X_y=True)
	X = normalize(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)
	rng = np.random.RandomState(42)
	random_unlabeled_points = rng.rand(len(X_train)) < 0.1
	y_train[random_unlabeled_points] = -1
	# 
	index, = np.where(y_train != -1)
	label_X_train = X_train[index, :]
	label_y_train = y_train[index]
	index, = np.where(y_train == -1)
	unlabel_X_train = X_train[index, :]
	unlabel_y = -1 * np.ones(unlabel_X_train.shape[0]).astype(int)
	return label_X_train, label_y_train, unlabel_X_train, unlabel_y, X_test, y_test


if __name__ == "__main__":
	label_X_train, label_y_train, unlabel_X_train, unlabel_y, X_test, y_test = get_data()
	DNN = DNN()
	DNNmodel = DNN.build_model(input_dim=30, output_dim=2)

	from semisupervised.PseudoLabelSSL import PseudoCallback, PseudoLabelNeuralNetworkClassifier
	pseudo_callback = PseudoCallback()

	model = PseudoLabelNeuralNetworkClassifier(DNNmodel, pseudo_callback)
	model.fit(np.vstack((label_X_train, unlabel_X_train)), np.append(label_y_train, unlabel_y))
	predict = model.predict(X_test)
	acc = metrics.accuracy_score(y_test, predict)
	print("pseudo-label accuracy", acc)



