# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:22:43 2020

@author: rosefun
"""
import keras.backend as K
from keras.callbacks import Callback
from keras.metrics import categorical_accuracy
from keras.layers.core import Dense, Activation, Dropout
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class PseudoCallback(Callback):
    def __init__(self, batch_size=128, update_epoch_loss=False, pretrain=True):
        self.batch_size = batch_size

        # unlabeled权重
        self.alpha_t = 0.0
        self.beta_t = 0.0
        self.best_test_acc = 0.0
        self.update_epoch_loss = update_epoch_loss
        self.pretrain = pretrain

    def train_generator(self, X, y):
        while True:
            n_batch = X.shape[0] // self.batch_size
            indices = np.arange(y.shape[0])
            np.random.shuffle(indices)
            for i in range(n_batch):
                current_indices = indices[i * self.batch_size: (i + 1) * self.batch_size]
                X_batch = X[current_indices]
                y_batch = y[current_indices]
                yield X_batch, y_batch

    def test_generator(self, X_test, y_test):
        while True:
            indices = np.arange(y_test.shape[0])
            for i in range(X_test.shape[0] // self.batch_size):
                current_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                X_batch = X_test[current_indices]
                y_batch = y_test[current_indices]
                yield X_batch, y_batch

    def crossentropy(self, y_true, y_pred):
        '''
        Input:
            y_true: np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
            y_pred: np.array([[0.8, 0.2, 0], [0.4, 0.5, 0.1], [0.1, 0.5, 0.4]])
        Output: np.array [0.22314355 0.69314718 0.69314718]
        '''
        # this gives the same result as using keras.objective.crossentropy
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss = -(K.sum(y_true * K.log(y_pred), axis=-1))

        return loss

    def make_loss(self):

        def loss_function(y_true, y_pred):
            """
            : y_true, 2-dim tensor, true label, the last columns is a flag whether the sample is the pseudo-labeled sample.
            : y_pred, 2-dim tensor.
            """

            if self.pretrain:
                y_true_item = y_true
            else:
                y_true_item = y_true[:, :-1]
            unlabeled_flag = K.reshape(y_true[:, -1], [-1])
            # the number of labeled samples and pseudo-labeled samples.

            labelIndex = K.tf.where(K.tf.equal(unlabeled_flag, 0.0))
            labelIndex = K.reshape(labelIndex, [-1])

            unlabelIndex = K.tf.where(K.tf.equal(unlabeled_flag, 1.0))
            unlabelIndex = K.reshape(unlabelIndex, [-1])

            labelY = K.gather(y_true_item, labelIndex)
            labelPredictY = K.gather(y_pred, labelIndex)
            unlabelY = K.gather(y_true_item, unlabelIndex)
            unlabelPredictY = K.gather(y_pred, unlabelIndex)

            loss = K.constant(0.0, dtype='float32')
            # alpha*CE(unlabelX, pseudo_label)
            loss += self.alpha_t * K.tf.cond(K.tf.equal(K.tf.size(unlabelY), 0), lambda: K.tf.constant(0.0),
                                             lambda: K.tf.reduce_mean(self.crossentropy(unlabelY, unlabelPredictY)))
            # CE(labeledX, labelY)
            loss += K.tf.cond(K.tf.equal(K.tf.size(labelY), 0), lambda: K.tf.constant(0.0),
                              lambda: K.tf.reduce_mean(self.crossentropy(labelY, labelPredictY)))
            return loss

        return loss_function

    def accuracy(self, y_true, y_pred):
        if self.pretrain:
            y_true_item = y_true
        else:
            y_true_item = y_true[:, :-1]
        return categorical_accuracy(y_true_item, y_pred)

    def on_epoch_end(self, epoch, logs):
        if self.update_epoch_loss:
            # alpha(t)の更新
            if epoch < 10:
                self.alpha_t = 0.0
                self.beta_t = 0.0
            elif epoch >= 70:
                self.alpha_t = 3
                self.beta_t = 0.0
            else:
                self.alpha_t = (epoch - 10.0) / (70.0 - 10.0) * 3
                self.beta_t = (epoch - 10.0) / (70.0 - 10.0) * 0.0

    def on_train_end(self, logs):
        """"""
        pass

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

class PseudoLabelNeuralNetworkClassifier(object):
    """
    """

    def __init__(self, clf, pseudo_callback, batch_size=128, pretrain_epoch=40, finetune_epoch=40):
        """
        :param clf:
        """

        self.model = clf
        self.finetune_model = keras.models.clone_model(clf)
        self.pseudo_callback = pseudo_callback
        self.batch_size = batch_size
        self.pretrain_epoch = pretrain_epoch
        self.finetune_epoch = finetune_epoch

    def onehot(self, narr, nclass=None):
        """
        :param narr: np.ndarray
        return onehot ndarray.
        """
        if not nclass:
            nclass = np.max(narr) + 1
        return np.eye(nclass)[narr]

    def fit(self, X, y):
        """
        :param X: numpy.ndarray, train datasets, 2-ndim
        :param y: numpy.ndarray, label of train datasets, scalar values, 1-ndim. 
                    If label == -1, the sample is unlabel.
        """
        unlabeledX = X[y == -1, :]  # .tolist()
        labeledX = X[y != -1, :]  # .tolist()
        labeled_y = y[y != -1]
        if labeled_y.ndim == 1:
            labeled_y = self.onehot(labeled_y)

        # step 1. train nn clf with labeled datasets.
        # pseudo_callback = PseudoCallback(batch_size=self.batch_size)
        self.model.compile(loss=pseudo_callback.make_loss(), optimizer='adam', metrics=[pseudo_callback.accuracy])
        clf = self.fit_model(labeledX, labeled_y, epochs=self.pretrain_epoch)

        # step 2. predict unlabeled dataset
        if len(unlabeledX) == 0:
            pass
        else:
            pseudo_label = self.predict(unlabeledX)

            pseudo_label = self.onehot(pseudo_label, np.max(y) + 1)
            # step 3. train clf with unlabeled datasets and labeled datasets.
            # add flag whether is pseudo-labeled sample
            labeled_y = np.hstack((labeled_y, np.zeros((len(labeled_y), 1))))
            pseudo_label = np.hstack((pseudo_label, -1 * np.ones((len(pseudo_label), 1))))
            # merge dataset
            merge_X_train = np.vstack((labeledX, unlabeledX))
            merge_y_train = np.vstack((labeled_y, pseudo_label))
            self.pseudo_callback.update_epoch_loss = True
            self.pseudo_callback.pretrain = False
            self.finetune_model.compile(loss=pseudo_callback.make_loss(), optimizer='adam',
                                        metrics=[pseudo_callback.accuracy])
            self.model = self.finetune_model
            clf = self.fit_model(merge_X_train, merge_y_train, epochs=self.finetune_epoch)
            self.clf = clf

    def fit_model(self, X_train, Y_train, X_test=None, Y_test=None, epochs=None):
        if X_test is not None:
            hist = self.model.fit_generator(self.pseudo_callback.train_generator(X_train, Y_train),
                                            steps_per_epoch=X_train.shape[0] // self.batch_size,
                                            validation_data=(X_test, Y_test), callbacks=[self.pseudo_callback],
                                            validation_steps=X_test.shape[0] // self.batch_size, epochs=epochs).history
        else:
            hist = self.model.fit_generator(self.pseudo_callback.train_generator(X_train, Y_train),
                                            steps_per_epoch=X_train.shape[0] // self.batch_size,
                                            callbacks=[self.pseudo_callback],
                                            epochs=epochs
                                            ).history

    def evaluate_model(self, X, Y):
        score = self.model.evaluate(X, Y, batch_size=self.batch_size, verbose=1)
        print('Test score:{:.4f}'.format(score[0]))
        print('Test accuracy:{:.4f}'.format(score[1]))
        return score[1]

    def predict_proba(self, X):
        pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        return pred

    def predict(self, X):
        """
        return 1-dim nd.array, scalar value of prediction. 
        """
        pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        pred = np.argmax(pred, axis=1)
        return pred

    """
    """

    def __init__(self, clf, pseudo_callback, batch_size=128, pretrain_epoch=40, finetune_epoch=40):
        """
        :param clf:
        """

        self.model = clf
        self.finetune_model = keras.models.clone_model(clf)
        self.pseudo_callback = pseudo_callback
        self.batch_size = batch_size
        self.pretrain_epoch = pretrain_epoch
        self.finetune_epoch = finetune_epoch

    def onehot(self, narr, nclass=None):
        """
        :param narr: np.ndarray
        return onehot ndarray.
        """
        if not nclass:
            nclass = np.max(narr) + 1
        return np.eye(nclass)[narr]

    def fit(self, X, y):
        """
        :param X: numpy.ndarray, train datasets, 2-ndim
        :param y: numpy.ndarray, label of train datasets, scalar values, 1-ndim. 
                    If label == -1, the sample is unlabel.
        """
        unlabeledX = X[y == -1, :]  # .tolist()
        labeledX = X[y != -1, :]  # .tolist()
        labeled_y = y[y != -1]
        if labeled_y.ndim == 1:
            labeled_y = self.onehot(labeled_y)

        # step 1. train nn clf with labeled datasets.
        # pseudo_callback = PseudoCallback(batch_size=self.batch_size)
        self.model.compile(loss=pseudo_callback.make_loss(), optimizer='adam', metrics=[pseudo_callback.accuracy])
        clf = self.fit_model(labeledX, labeled_y, epochs=self.pretrain_epoch)

        # step 2. predict unlabeled dataset
        if len(unlabeledX) == 0:
            pass
        else:
            pseudo_label = self.predict(unlabeledX)

            pseudo_label = self.onehot(pseudo_label, np.max(y) + 1)
            # step 3. train clf with unlabeled datasets and labeled datasets.
            # add flag whether is pseudo-labeled sample
            labeled_y = np.hstack((labeled_y, np.zeros((len(labeled_y), 1))))
            pseudo_label = np.hstack((pseudo_label, -1 * np.ones((len(pseudo_label), 1))))
            # merge dataset
            merge_X_train = np.vstack((labeledX, unlabeledX))
            merge_y_train = np.vstack((labeled_y, pseudo_label))
            self.pseudo_callback.update_epoch_loss = True
            self.pseudo_callback.pretrain = False
            self.finetune_model.compile(loss=pseudo_callback.make_loss(), optimizer='adam',
                                        metrics=[pseudo_callback.accuracy])
            self.model = self.finetune_model
            clf = self.fit_model(merge_X_train, merge_y_train, epochs=self.finetune_epoch)
            self.clf = clf

    def fit_model(self, X_train, Y_train, X_test=None, Y_test=None, epochs=None):
        if X_test is not None:
            hist = self.model.fit_generator(self.pseudo_callback.train_generator(X_train, Y_train),
                                            steps_per_epoch=X_train.shape[0] // self.batch_size,
                                            validation_data=(X_test, Y_test), callbacks=[self.pseudo_callback],
                                            validation_steps=X_test.shape[0] // self.batch_size, epochs=epochs).history
        else:
            hist = self.model.fit_generator(self.pseudo_callback.train_generator(X_train, Y_train),
                                            steps_per_epoch=X_train.shape[0] // self.batch_size,
                                            callbacks=[self.pseudo_callback],
                                            epochs=epochs
                                            ).history

    def evaluate_model(self, X, Y):
        score = self.model.evaluate(X, Y, batch_size=self.batch_size, verbose=1)
        print('Test score:{:.4f}'.format(score[0]))
        print('Test accuracy:{:.4f}'.format(score[1]))
        return score[1]

    def predict_proba(self, X):
        pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        return pred

    def predict(self, X):
        """
        return 1-dim nd.array, scalar value of prediction. 
        """
        pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        pred = np.argmax(pred, axis=1)
        return pred

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




