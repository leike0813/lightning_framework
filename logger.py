import os
import datetime
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from .default_config import DEFAULT_CONFIG


def build_logger(config=DEFAULT_CONFIG):
    if config.LOGGER.NAME == 'MLFlowLogger':
        logger = MLFlowLogger(
            tracking_uri=config.LOGGER.MLFLOW.tracking_uri,
            artifact_location=os.path.join(config.ENVIRONMENT.MLFLOW_BASE_PATH, config.LOGGER.MLFLOW.artifact_location),
            experiment_name=config.LOGGER.EXPERIMENT_NAME,
            run_name='{abbr}{spec}{tag}_{time}'.format(
                abbr=config.MODEL.ABBR[config.MODEL.TYPE] + '_' if hasattr(config.MODEL.ABBR, config.MODEL.TYPE) else '',
                spec=config.MODEL.SPEC_NAME + '_' if config.MODEL.SPEC_NAME != '' else '',
                tag=config.TAG,
                time=datetime.datetime.now().strftime('%d%H%M')
            ),
            tags=config.LOGGER.TAGS,
            log_model=config.LOGGER.LOG_MODEL,
        )
    else:
        logger = None

    return logger