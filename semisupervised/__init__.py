from __future__ import absolute_import 
"""
The :mod:`.semi_supervised` module implements semi-supervised learning
algorithms. These algorithms utilized small amounts of labeled data and large
amounts of unlabeled data for classification tasks. This module includes Label
Propagation.
"""
# import os
from .qns3vm import QN_S3VM
from .TSVM import SKTSVM
# #from . import qns3vm

__all__ = ['QN_S3VM','SKTSVM']