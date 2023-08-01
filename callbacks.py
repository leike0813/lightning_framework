
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from .loss_logger import LogLossCallback
from .default_config import DEFAULT_CONFIG


def build_callbacks(config=DEFAULT_CONFIG):
    callbacks = []
    hparams = {
        'early-stopping': False,
        'checkpointing': False,
    }
    if config.TRAIN.LOG_LEARNINGRATE:
        callbacks.append(LearningRateMonitor())
    if config.TRAIN.LOG_LOSS:
        callbacks.append(LogLossCallback())
    if config.TRAIN.USE_EARLYSTOPPING:
        callbacks.append(L.pytorch.callbacks.EarlyStopping(
            monitor=config.TRAIN.MONITOR,
            mode=config.TRAIN.MONITOR_MODE,
            patience=config.TRAIN.EARLYSTOPPING_PATIENCE,
        ))
        hparams['early-stopping'] = True
        hparams.update({
            'early-stopping-patience': config.TRAIN.EARLYSTOPPING_PATIENCE,
            'train-monitor': config.TRAIN.MONITOR
        })
    if config.TRAIN.USE_CHECKPOINT:
        callbacks.append(L.pytorch.callbacks.ModelCheckpoint(
            monitor=config.TRAIN.MONITOR,
            mode=config.TRAIN.MONITOR_MODE,
            save_top_k=config.TRAIN.CHECKPOINT_TOPK,
            save_last=config.TRAIN.CHECKPOINT_SAVELAST,
        ))
        hparams['checkpointing'] = True
        hparams.update({
            'train-monitor': config.TRAIN.MONITOR
        })

    return callbacks, hparams
