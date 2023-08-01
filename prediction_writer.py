import os
import warnings
from pathlib import Path
from collections import Sequence
from PIL import Image as PILImage
import torch
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import BasePredictionWriter
from .default_config import DEFAULT_CONFIG


__all__ = [
    'build_prediction_writer',
    'PredictionWriter',
]


def build_prediction_writer(config=DEFAULT_CONFIG):
        return PredictionWriter(
            output_dir=os.path.join(config.ENVIRONMENT.RESULT_BASE_PATH, config.PREDICT.RESULT_PATH),
            **config.PREDICT.WRITER
        )


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, image_suffixs='', image_format='png', concatenate=False, log_prediction=False, log_folder='', write_interval='batch'):
        super(PredictionWriter, self).__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.image_suffixs = image_suffixs if isinstance(image_suffixs, Sequence) else [image_suffixs]
        self.image_format = image_format
        if concatenate == 0:
            warnings.warn('Concatenate in dimension 0 means to concatenate image channels, which is confusing. The concatenation will not be applied.', UserWarning)
        elif concatenate > 2:
            raise ValueError('Concatenate in dimension higher than 2 is not allowed.')
        self.concatenate = concatenate
        self.log_prediction = log_prediction
        self.log_folder = log_folder

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        # img_names: list of length B(batch_size)
        # pred: list of tensors of length N(number of predict output), tensors are of shape B, C, H, W
        pred, img_names= prediction
        # paths: nested list of paths (N x B, if not concatenated) or list of paths (length B, otherwise)
        paths = self.get_paths(img_names)
        # transformed pred: nested list of PIL images(N x B, if not concatenated) or list of PIL images(length B, otherwise)
        pred = self.transform(pred)
        self.save_image(pred, paths, pl_module)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)
        for (pred, img_names) in predictions:
            paths = self.get_paths(img_names)
            pred = self.transform(pred)
            self.save_image(pred, paths, pl_module)

    def get_paths(self, img_names):
        paths = []
        if not self.concatenate:
            for suffix in self.image_suffixs:
                suffix_paths = []
                for i in range(len(img_names)):
                    suffix_paths.append(self.output_dir / '{img}{suf}.{fmt}'.format(
                        img=img_names[i],
                        suf = '_' + suffix if suffix != '' else '',
                        fmt = self.image_format
                    ))
                paths.append(suffix_paths)
        else:
            for i in range(len(img_names)):
                paths.append(self.output_dir / '{img}_concat.{fmt}'.format(
                        img=img_names[i],
                        fmt = self.image_format
                    ))

        return paths

    def transform(self, pred):
        ret = []
        if not self.concatenate:
            for i in range(len(self.image_suffixs)):
                pred_img = pred[i]
                mode = 'RGB' if pred_img.shape[1] == 3 else 'L'
                suffix_imgs = []
                for j in range(pred_img.shape[0]):
                    suffix_imgs.append(TF.to_pil_image(pred_img[j], mode=mode))
                ret.append(suffix_imgs)
        else:
            pred = torch.cat(pred, dim=self.concatenate + 1)
            mode = 'RGB' if pred.shape[1] == 3 else 'L'
            for j in range(pred.shape[0]):
                ret.append(TF.to_pil_image(pred[j], mode=mode))

        return ret

    def save_image(self, pred, paths, pl_module):
        if not self.concatenate:
            for i in range(len(self.image_suffixs)):
                suffix_paths = paths[i]
                for j in range(len(suffix_paths)):
                    pred[i][j].save(suffix_paths[j])
                    if self.log_prediction and hasattr(pl_module.logger.experiment, 'log_artifact'):
                        pl_module.logger.experiment.log_artifact(
                            pl_module.logger._run_id,
                            suffix_paths[j],
                            self.log_folder,
                        )
        else:
            for i in range(len(paths)):
                pred[i].save(paths[i])
                if self.log_prediction and hasattr(pl_module.logger.experiment, 'log_artifact'):
                    pl_module.logger.experiment.log_artifact(
                        pl_module.logger._run_id,
                        paths[i],
                        self.log_folder,
                    )