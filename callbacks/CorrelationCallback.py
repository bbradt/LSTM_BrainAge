from typing import Iterable, Union

import torch

from catalyst.metrics import FunctionalBatchMetric
from catalyst.callbacks.metric import FunctionalBatchMetricCallback

def correlation(outputs, targets):
    stacked = torch.stack([outputs, targets], 0).squeeze()
    corr = torch.corrcoef(stacked).flatten()[1]
    return corr

class CorrelationCallback(FunctionalBatchMetricCallback):
    def __init__(self, input_key, target_key):
        super(CorrelationCallback, self).__init__(
            FunctionalBatchMetric(metric_fn=correlation, metric_key="correlation"),input_key, target_key)