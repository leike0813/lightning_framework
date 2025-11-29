import os
from pathlib import Path
import numpy as np
import torch
from torchmetrics import ROC
from torchmetrics.utilities.enums import ClassificationTask
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.mlflow import MLFlowLogger
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use('Agg')
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


class ROCCallback(Callback):
    def __init__(self, task, num_classes, category='defualt', append_category_name=False,
                 thresholds=1000, figsize=(3.15, 3.15), dpi=600):
        super().__init__()
        task = ClassificationTask.from_str(task)
        self.task = task
        self.num_classes = num_classes
        self.thresholds = thresholds
        self.category = category
        self.append_category_name = append_category_name
        self.figsize = figsize
        self.dpi = dpi
        self.__METRIC_MOVED = False

        if _MATPLOTLIB_AVAILABLE:
            try:
                import scienceplots
                plt.style.use(['science', 'no-latex'] if num_classes > 3 else ['science', 'high-contrast', 'no-latex'])
            except ImportError:
                pass

    @property
    def state_key(self):
        return f'{self.category}_ROC'

    def state_dict(self):
        return {
            'task': self.task,
            'num_classes': self.num_classes,
            'category': self.category,
            'append_category_name': self.append_category_name,
        }

    def setup(self, trainer, pl_module, stage):
        if stage == 'test':
            self.metric = ROC(task=self.task, num_classes=self.num_classes, thresholds=self.thresholds)

    def on_test_start(self, trainer, pl_module):
        self._data_fetched = False
        if not self.__METRIC_MOVED:
            self.metric.to(pl_module.device)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        input = self.fetch_data_for_calc(outputs)
        if input is not None:
            self._data_fetched = True
            self.metric(*input)

    def on_test_epoch_end(self, trainer, pl_module):
        if self._data_fetched:
            fpr, tpr, thresh = self.metric.compute()
            self.save_and_log_artifact(trainer, fpr, 'FPR')
            self.save_and_log_artifact(trainer, tpr, 'TPR')
            self.save_and_log_artifact(trainer, thresh, 'Thresholds')
            if _MATPLOTLIB_AVAILABLE:
                fig = self.plot_roc()
                self.save_and_log_artifact(trainer, fig, 'ROC')
            self.metric.reset()
        self._data_fetched = False

    def fetch_data_for_calc(self, outputs):
        keys = ['metric_{cat}'.format(cat=self.category), 'metrics_{cat}'.format(cat=self.category), 'metric_src_{cat}'.format(cat=self.category), self.category]
        if self.category == 'default':
            keys.extend(['metric', 'metrics', 'metric_src'])
        for key in keys:
            data = outputs.get(key)
            if data is not None:
                return data
        return None

    def plot_roc(self):
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.gca()
        _, ax = self.metric.plot(score=True, ax=ax)
        return fig

    @staticmethod
    def get_run_basefld(trainer):
        if trainer.checkpoint_callback:
            _run_basefld = Path(trainer.checkpoint_callback.dirpath).parent
            return _run_basefld if _run_basefld.name == trainer.logger.run_id else None
        else:
            return (Path(trainer.default_root_dir) / trainer.logger.experiment_id) / trainer.logger.run_id

    def save_figure(self, fig, name, local_fld_path):
        local_filename = 'test_epoch_{nme}{cat}.png'.format(
            nme=name,
            cat='_' + self.category if self.append_category_name else ''
        )
        local_filepath = Path(local_fld_path) / local_filename
        fig.savefig(local_filepath)
        return local_filepath

    def save_tensor(self, ten, name, local_fld_path):
        local_filename = 'test_epoch_{nme}{cat}.pt'.format(
            nme=name,
            cat='_' + self.category if self.append_category_name else ''
        )
        local_filepath = Path(local_fld_path) / local_filename
        torch.save(ten, local_filepath)
        return local_filepath

    def save_and_log_artifact(self, trainer, obj, name, artifact_path='metrics'):
        if isinstance(trainer.logger, MLFlowLogger):
            run_base_fld = self.get_run_basefld(trainer)
            if isinstance(obj, mpl.figure.Figure):
                local_filepath = self.save_figure(obj, name, run_base_fld)
            elif isinstance(obj, torch.Tensor):
                local_filepath = self.save_tensor(obj, name, run_base_fld)

            trainer.logger.experiment.log_artifact(
                trainer.logger.run_id,
                local_filepath.as_posix(),
                artifact_path
            )
            os.remove(local_filepath)

