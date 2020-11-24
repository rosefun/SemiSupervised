from __future__ import absolute_import 
"""
The :mod:`.semi_supervised` module implements semi-supervised learning
algorithms. These algorithms utilized small amounts of labeled data and large
amounts of unlabeled data for classification tasks. This module includes Label
Propagation.
"""

from .qns3vm import QN_S3VM
from .TSVM import S3VM
from .SklearnLabelPropagation import LabelPropagation
try:
	from .StackedAutoEncoderSSL import StackedAutoEncoderClassifier
	module = ['QN_S3VM', 'S3VM', 'LabelPropagation', 'StackedAutoEncoderClassifier']
except Exception as e:
	print("Exception:", e)
	print("We recommend you install torch first.")
	module = ['QN_S3VM', 'S3VM', 'LabelPropagation']
__all__ = module