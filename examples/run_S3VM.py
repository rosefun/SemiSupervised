from __future__ import absolute_import
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

import os
os.chdir(r"F:\研究生\github项目\python半监督包\semisupervised")
import sys
sys.path.append(r"F:\研究生\github项目\python半监督包\semisupervised")

# normalization
def normalize(x):
	return (x - np.min(x))/(np.max(x) - np.min(x))

def get_data():
	X, y = datasets.load_breast_cancer(return_X_y=True)
	X = normalize(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6, random_state = 0)
	rng = np.random.RandomState(42)
	random_unlabeled_points = rng.rand(len(X_train)) < 0.1
	y_train[random_unlabeled_points] = -1
	# 
	index, = np.where(y_train != -1)
	label_X_train = X_train[index,:]
	label_y_train = y_train[index]
	index, = np.where(y_train == -1)
	unlabel_X_train = X_train[index,:]
	unlabel_y = -1*np.ones(unlabel_X_train.shape[0]).astype(int)
	return label_X_train, label_y_train, unlabel_X_train, unlabel_y, X_test, y_test

if __name__ == "__main__":
	from semisupervised import S3VM
	
	label_X_train, label_y_train, unlabel_X_train, unlabel_y, X_test, y_test = get_data()
	# S3VM
	model = S3VM()
	model.fit(np.vstack((label_X_train,unlabel_X_train)), np.append(label_y_train, unlabel_y))
	predict = model.predict(X_test)
	acc = metrics.accuracy_score(y_test, predict)
	print("S3VM accuracy", acc)
