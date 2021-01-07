# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator


class StackedAutoEncoderClassifier(BaseEstimator):
    def __init__(self, SAE, pretrain_epochs=100, finetune_epochs=100,
                 pretrain_optimizer_parameters=dict(lr=0.003, weight_decay=1e-5),
                 finetune_optimizer_parameters=dict(lr=0.003),
                 pretrain_batch_size=256, finetune_batch_size=256, pretrain_optimizer=None,
                 finetune_optimizer=None, patience=40,
                 device_name="auto", verbose=1, save_pretrain_model=False):
        self._estimator_type = "classifier"
        self.classes_ = None
        if device_name == 'auto' or not device_name:
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)
        if verbose:
            print("Info:", f"Device used : {self.device}")

        self.SAE = SAE.to(self.device)
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs

        self.pretrain_batch_size = pretrain_batch_size
        self.finetune_batch_size = finetune_batch_size
        if pretrain_optimizer is None:
            self.pretrain_optimizer = torch.optim.Adam(SAE.parameters(), **pretrain_optimizer_parameters)
        else:
            self.pretrain_optimizer = pretrain_optimizer
        if finetune_optimizer is None:
            self.finetune_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, SAE.parameters()),
                                                       **finetune_optimizer_parameters)
        else:
            self.finetune_optimizer = finetune_optimizer

        self.patience = patience
        self.verbose = verbose
        self.save_pretrain_model = save_pretrain_model
        self.pretrain_loss = nn.MSELoss()
        self.finetune_loss = nn.CrossEntropyLoss()

    def create_dataloader(self, X, y=None, batch_size=256, shuffle=False, device="cuda"):
        """
        Return: DataLoader of tensor data.
        """
        X = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float).to(self.device)
        if y is not None:
            y = torch.tensor(y, dtype=torch.float).to(self.device)
            tensor_data = TensorDataset(X, y)
        else:
            tensor_data = TensorDataset(X)
        dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def onehot(self, narr, nclass=None):
        """
        :param narr: np.ndarray
        return onehot ndarray.
        """
        if not nclass:
            nclass = np.max(narr) + 1
        return np.eye(nclass)[narr]

    def save_model(self, ):
        """save pretrained model.
        """
        checkpoint = {'model': self.SAE,
                      'state_dict': self.SAE.state_dict(),
                      'pretrain_optimizer': self.pretrain_optimizer.state_dict()}
        ckpt_file = 'SAE_pretrain.pth'
        torch.save(checkpoint, ckpt_file)
        return ckpt_file

    def fit(self, X, y, is_pretrain=True, validation_data=None):
        """
        :param X: np.ndarray
        :param y: 1-dim np.ndarray, scalar value, if value == -1, it means unlabeled sample.
        """
        # process data
        unlabeledX = X[y == -1].values
        labeledX = X[y != -1].values
        nclass = np.max(y) + 1
        self.classes_ = sorted(y[y != -1].unique())
        labeled_y = y[y != -1]
        if labeled_y.ndim == 1:
            labeled_y = self.onehot(labeled_y)

        if is_pretrain:
            # step 1. pretrain
            train_loader = self.create_dataloader(X, batch_size=self.pretrain_batch_size, shuffle=True)
            min_loss = float('inf')
            patience_counter = 0
            for epoch in range(self.pretrain_epochs):
                # train
                self.SAE.train()
                p_loss = 0.0
                for i, (x_batch,) in enumerate(train_loader):
                    x_batch = Variable(x_batch).to(self.device)
                    # forward
                    encoder_x, output = self.SAE(x_batch)
                    loss = self.pretrain_loss(output, x_batch)
                    # ===================backward====================
                    self.pretrain_optimizer.zero_grad()
                    loss.backward()
                    p_loss += loss
                    self.pretrain_optimizer.step()
                # ===================log========================
                p_loss = p_loss.item() / (i + 1)
                if p_loss <= min_loss:
                    patience_counter = 0
                    min_loss = p_loss
                else:
                    patience_counter += 1
                if self.verbose and (epoch % self.verbose == 0):
                    print('Info: epoch [{}/{}], loss:{:.4f}'
                          .format(epoch + 1, self.pretrain_epochs, p_loss))
                if patience_counter >= self.patience:
                    break

            if self.save_pretrain_model:
                self.save_model()

        # step 2. finetune 
        self.SAE.train()
        train_loader = self.create_dataloader(labeledX, labeled_y, batch_size=self.finetune_batch_size)
        if validation_data is not None:
            if validation_data[1].ndim == 1:
                valid_y = self.onehot(validation_data[1], nclass)
            valid_loader = self.create_dataloader(validation_data[0], y=valid_y)
        min_loss = float('inf')
        patience_counter = 0
        for ep in range(self.finetune_epochs):
            self.SAE.train()
            e_loss = 0
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = Variable(x_batch).to(self.device)
                prediction, reconstruct = self.SAE(x_batch)
                # cross_entropy loss, (input,target)
                loss = self.finetune_loss(prediction, torch.argmax(y_batch, dim=1))
                self.finetune_optimizer.zero_grad()
                loss.backward()
                self.finetune_optimizer.step()
                e_loss += loss
            e_loss = e_loss.item() / (i + 1)
            if validation_data is not None:
                valid_loss = self.predict_epoch(valid_loader)
                if valid_loss <= min_loss:
                    patience_counter = 0
                    min_loss = valid_loss
                else:
                    patience_counter += 1
                if self.verbose and (ep % self.verbose == 0):
                    print("Info: epoch:{}, loss:{:.4}, valid_loss:{:.4}".format(ep, e_loss, valid_loss))
                if patience_counter >= self.patience:
                    break
            else:
                if self.verbose and (ep % self.verbose == 0):
                    print("Info: epoch:{},loss:{:.4}".format(ep, e_loss))
        return self.SAE 
        
    def predict_epoch(self, valid_loader):
        self.SAE.eval()
        e_loss = 0
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            x_batch = Variable(x_batch).to(self.device)
            prediction, reconstruct = self.SAE(x_batch)
            # cross_entropy loss, (input,target)
            loss = self.finetune_loss(prediction, torch.argmax(y_batch, dim=1))
            e_loss += loss
        valid_loss = e_loss.item() / (i + 1)
        return valid_loss

    def predict(self, X_test):
        """
        """
        test_preds = self.predict_proba(X_test)
        pred = np.argmax(test_preds, axis=1)
        return pred

    def predict_proba(self, X_test):
        """
        return np.ndarray.
        """
        # eval test set
        test_preds = []
        test_loader = self.create_dataloader(X_test, shuffle=False)
        self.SAE.eval()
        for i, (x_batch,) in enumerate(test_loader):
            y_pred, reconstruct = self.SAE(x_batch)
            y_pred = y_pred.detach()
            test_preds.append(y_pred.cpu().numpy())
        test_preds = np.vstack(test_preds)
        return test_preds
    
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)