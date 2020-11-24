# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from torch import nn

from semisupervised import StackedAutoEncoderClassifier
from examples.example_utils import get_data


class StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, encoder=None, decoder=None):
        super(StackedAutoEncoder, self).__init__()
        if encoder is None:
            assert input_dim is not None, "The input feature dimension should be inputed"
            assert output_dim is not None, "The number of classes should be added."
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 100),
                nn.ReLU(True),
                nn.Linear(100, 50),
                nn.ReLU(True),
                nn.Linear(50, output_dim))
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = nn.Sequential(
                nn.Linear(output_dim, 50),
                nn.ReLU(True),
                nn.Linear(50, 100),
                nn.ReLU(True),
                nn.Linear(100, input_dim),
                nn.ReLU(True))
        else:
            self.decoder = decoder

    def init_parameters(self):
        """
        Some methods to initialize the paramethers of network
        Usage: network.apply(init_parameters)
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in
              name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in
              name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        """
        return encoder_x: hidden variable
        re_x: reconstruct input
        """
        encoder_x = self.encoder(x)
        reconstruct_x = self.decoder(encoder_x)
        return encoder_x, reconstruct_x


if __name__ == "__main__":
    label_X_train, label_y_train, unlabel_X_train, unlabel_y, X_test, y_test = get_data()
    X = np.vstack((label_X_train, unlabel_X_train))
    y = np.append(label_y_train, unlabel_y)
    SAE = StackedAutoEncoder(30, 2)
    SAE_clf = StackedAutoEncoderClassifier(SAE, pretrain_epochs=100, finetune_epochs=200,
                                           pretrain_optimizer_parameters=dict(lr=0.003, weight_decay=1e-5),
                                           finetune_optimizer_parameters=dict(lr=0.003),
                                           pretrain_batch_size=256, finetune_batch_size=256, pretrain_optimizer=None,
                                           finetune_optimizer=None, patience=40,
                                           device_name="auto", verbose=1, save_pretrain_model=False)
    SAE_clf.fit(X, y, is_pretrain=True, validation_data=(X_test, y_test))
    predict = SAE_clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, predict)
    print("SAE acc", acc)