from typing import Any, Callable, List, Optional, Tuple, Type, Union, Sequence

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_compute,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
)
from torchmetrics.functional.classification.accuracy import _accuracy_reduce
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["BinaryAccuracy.plot", "MulticlassAccuracy.plot", "MultilabelAccuracy.plot"]


class BinaryAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    tp: Union[List[Tensor], Tensor]
    fp: Union[List[Tensor], Tensor]
    tn: Union[List[Tensor], Tensor]
    fn: Union[List[Tensor], Tensor]

    def __init__(
        self,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        zero_division = kwargs.pop("zero_division", 0)
        super().__init__(**kwargs)
        if validate_args:
            _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index, zero_division)
        self.threshold = threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self._create_state(size=1, multidim_average=multidim_average)

    def _create_state(
        self,
        size: int,
        multidim_average: Literal["global", "samplewise"] = "global",
    ) -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == "samplewise":
            default = list
            dist_reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = "sum"

        self.add_state("tp", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("fp", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("tn", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("fn", default(), dist_reduce_fx=dist_reduce_fx)

    def _update_state(self, tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> None:
        """Update states depending on multidim_average argument."""
        if self.multidim_average == "samplewise":
            self.tp.append(tp)  # type: ignore[union-attr]
            self.fp.append(fp)  # type: ignore[union-attr]
            self.tn.append(tn)  # type: ignore[union-attr]
            self.fn.append(fn)  # type: ignore[union-attr]
        else:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn

    def _final_state(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Aggregate states that are lists and return final states."""
        tp = dim_zero_cat(self.tp)
        fp = dim_zero_cat(self.fp)
        tn = dim_zero_cat(self.tn)
        fn = dim_zero_cat(self.fn)
        return tp, fp, tn, fn

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _binary_stat_scores_tensor_validation(preds, target, self.multidim_average, self.ignore_index)
        preds, target = _binary_stat_scores_format(preds, target, self.threshold, self.ignore_index)
        tp, fp, tn, fn = _binary_stat_scores_update(preds, target, self.multidim_average)
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:

        return self._plot(val, ax)