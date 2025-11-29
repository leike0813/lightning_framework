import os
import warnings
from pathlib import Path
from datetime import datetime
from collections.abc import Sequence
from typing import Any, Optional

import torch
import numpy as np
import pandas as pd
from pytorch_framework.utils import CustomCfgNode as CN
from lightning.pytorch.callbacks import BasePredictionWriter



__all__ = [
    'CSVPredictionWriter',
]


class CSVPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, columns=None, log_prediction=False, log_folder=''):
        super().__init__(write_interval="batch_and_epoch")
        self.output_dir = Path(output_dir)
        self.columns = columns
        self.log_prediction = log_prediction
        self.log_folder = log_folder
        self.columns = columns
        if columns is not None:
            self.df = pd.DataFrame(columns=columns)
        else:
            self.df = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if self.df is None:
            columns = getattr(pl_module, 'output_columns', None)
            if columns:
                self.df = pd.DataFrame(columns=columns)
                self.columns = columns

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
        if self.df is None:
            warnings.warn("No columns defined for the dataframe, skipping prediction writing.", UserWarning)
        else:
            _df = pd.DataFrame(data=prediction, columns=self.columns)
            self.df = pd.concat([self.df, _df], ignore_index=True)

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        if self.df is None:
            warnings.warn("No columns defined for the dataframe, export terminated.", UserWarning)
        else:
            if not self.output_dir.exists():
                os.makedirs(output_dir)

            _logging_available = False
            if self.log_prediction and hasattr(pl_module.logger.experiment, 'log_artifact'):
                _logging_available = True

            output_id = getattr(getattr(pl_module, 'logger', None), 'run_id', datetime.now().strftime("%y%m%d%H%M"))
            file_path = self.output_dir / f'{output_id}_output.csv'
            self.df.to_csv(file_path, index=False)
            if _logging_available:
                pl_module.logger.experiment.log_artifact(
                    pl_module.logger._run_id,
                    file_path,
                    log_folder,
                )