import os
import torch
from lightning import Trainer
from .default_config import DEFAULT_CONFIG


def build_trainer(config=DEFAULT_CONFIG, logger=None, prediction_writer=None, callbacks=[]):
    hparams = {
        'precision': config.TRAIN.TRAINER.precision,
        'min-epochs': config.TRAIN.TRAINER.min_epochs,
        'max-epochs': config.TRAIN.TRAINER.max_epochs,
        'overfit-batches': config.TRAIN.TRAINER.overfit_batches,
    }
    if prediction_writer is not None:
        callbacks.append(prediction_writer)
    if config.TRAIN.TRAINER.precision in ['16-mixed', 'bf16-mixed', '16', 'bf16', 16]:
        torch.set_float32_matmul_precision('high')

    if os.path.isfile(os.path.join(config.ENVIRONMENT.DATA_BASE_PATH, config.TRAIN.CKPT_PATH)):
        hparams.update({
            'checkpoint': os.path.join(config.ENVIRONMENT.DATA_BASE_PATH, config.TRAIN.CKPT_PATH)
        })
    return (
        Trainer(
            logger=logger,
            default_root_dir=config.ENVIRONMENT.RESULT_BASE_PATH,
            callbacks=callbacks,
            **config.TRAIN.TRAINER
        ),
        hparams
    )
