# SemiSupervised



# Describe



This is a Semi-supervised learning framework of Python. You can use it for classification task in machine learning.

 

# Install



```
pip install semisupervised
```





# API



We have implemented following semi-supervised learning algorithm. 



All the methods are similar to Sklearn Semi-supervised API. 



Step 1. **The unlabeled samples should be labeled as -1.**

Step2. `model.fit(X,y)`

Step3. `model.predict(X_test)`





- LabelPropagation

   [reference code](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html#sklearn.semi_supervised.LabelPropagation)

- S3VM 

  [reference code](<https://github.com/d12306/Implementation-of-Transductive-SVM-Sklearn-Compatible>)

- PseudoLabelNeuralNetworkClassifier

  This is pseudo-label semi-supervised learning, `PseudoLabelNeuralNetworkClassifier`  should work with `PseudoCallback` .

  



Statement

Some of the code comes from the Internet.

## Examples



```python
from __future__ import absolute_import
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

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

label_X_train, label_y_train, unlabel_X_train, unlabel_y, X_test, y_test = get_data()

# import 
from semisupervised import S3SVM

model = S3SVM()
model.fit(np.vstack((label_X_train, unlabel_X_train)), np.append(label_y_train, unlabel_y))
# predict
predict = model.predict(X_test)
acc = metrics.accuracy_score(y_test, predict)
# metric
print("accuracy", acc)
```



Please see `examples`  folder for more examples.

 