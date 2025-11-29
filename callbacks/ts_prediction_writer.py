import os
import warnings
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Optional

import torch
import numpy as np
from pytorch_framework.utils import CustomCfgNode as CN
from lightning.pytorch.callbacks import BasePredictionWriter


__all__ = [
    'TSPredictionWriter',
]


class TSPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, log_prediction=False, log_folder='', write_interval="epoch"):
        super().__init__(write_interval)
        if write_interval == 'batch_and_epoch':
            warnings.warn("Write on batch and epoch simutaniously is unnecessary, and doing so will cause additional I/O overhead.")
        self.output_dir = Path(output_dir)
        self.log_prediction = log_prediction
        self.log_folder = log_folder

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.interval.on_batch():
            pred, file_names = prediction
            self.write_prediction(pred, file_names, self.output_dir, self.log_prediction, self.log_folder, pl_module)

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Any,
        batch_indices: Optional[Sequence[int]],
    ) -> None:
        if self.interval.on_epoch():
            for (pred, file_names) in predictions:
                self.write_prediction(pred, file_names, self.output_dir, self.log_prediction, self.log_folder, pl_module)

    @staticmethod
    def write_prediction(pred, file_names, output_dir, log_prediction, log_folder, pl_module):
        if not output_dir.exists():
            os.makedirs(output_dir)

        _logging_available = False
        if log_prediction and hasattr(pl_module.logger.experiment, 'log_artifact'):
            _logging_available = True

        result_paths = []
        for suffix, data in pred.items():
            assert isinstance(data, torch.Tensor) and data.ndim == 3, 'Data in prediction must be torch.Tensor with ndim = 3'
            assert data.shape[0] == len(file_names), 'Unmatched number of file names.'

            for i in range(len(file_names)):
                file_path = output_dir / '{nme}{suf}.npy'.format(
                    nme=file_names[i], suf='_' + suffix if suffix != '' else ''
                )
                data_numpy = data[i].detach().cpu().numpy()
                np.save(file_path, data_numpy)
                result_paths.append(file_path)
                if _logging_available:
                    pl_module.logger.experiment.log_artifact(
                        pl_module.logger._run_id,
                        file_path,
                        log_folder,
                    )

        return result_paths