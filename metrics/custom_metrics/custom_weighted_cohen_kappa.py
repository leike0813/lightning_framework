from typing import Any, Optional, Union
from typing_extensions import Literal

import torch
from torch import Tensor
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.metric import Metric
from torchmetrics.functional.classification.confusion_matrix import (

    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
)


class CustomWeightedCohenKappa(MulticlassConfusionMatrix):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
            self,
            num_classes: int,
            weights: Union[list, tuple],
            ignore_index: Optional[int] = None,
            validate_args: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(num_classes, ignore_index, normalize=None, validate_args=False, **kwargs)
        if validate_args:
            _custom_cohen_kappa_arg_validation(num_classes, weights, ignore_index)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        confmat = self.confmat.float() if not self.confmat.is_floating_point() else self.confmat
        n_classes = confmat.shape[0]
        sum0 = confmat.sum(dim=0, keepdim=True)
        sum1 = confmat.sum(dim=1, keepdim=True)
        expected = sum1 @ sum0 / sum0.sum()  # outer product

        w_mat = torch.zeros_like(confmat)
        w_mat += self.weights
        w_mat = torch.abs(w_mat - w_mat.T)

        k = torch.sum(w_mat * confmat) / torch.sum(w_mat * expected)
        return 1 - k


def _custom_cohen_kappa_arg_validation(
    num_classes: int,
    weights: Union[list, tuple],
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``weights`` has to be a list or tuple with length equal to num_classes
    - ``ignore_index`` has to be None or int
    """
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize=None)
    if not isinstance(weights, (list, tuple)) or len(weights) != num_classes:
        raise ValueError(f"Expected argument `weight` to be a list or tuple with length of {num_classes}.")