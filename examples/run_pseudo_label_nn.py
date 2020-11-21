#!/usr/bin/env python
# encoding=utf-8
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


if __name__ == "__main__":
    label_X_train, label_y_train, unlabel_X_train, unlabel_y, X_test, y_test = get_data()
    DNN = DNN()
    DNNmodel = DNN.build_model(input_dim=30, output_dim=2)

    from semisupervised.PseudoLabelSSL import PseudoCallback, PseudoLabelNeuralNetworkClassifier
    pseudo_callback = PseudoCallback()
    clf = PseudoLabelNeuralNetworkClassifier(DNNmodel, pseudo_callback)
    clf.fit(np.vstack((label_X_train, unlabel_X_train)), np.append(label_y_train, unlabel_y))



