from pytorch_framework.utils import CustomCfgNode as CN


DEFAULT_CONFIG = CN()

# Base config files
DEFAULT_CONFIG.BASE = ['']

# Environment settings
DEFAULT_CONFIG.ENVIRONMENT = CN(visible=False)
DEFAULT_CONFIG.ENVIRONMENT.PROJECT_PATH = ''
DEFAULT_CONFIG.ENVIRONMENT.DATA_BASE_PATH = 'data'
DEFAULT_CONFIG.ENVIRONMENT.RESULT_BASE_PATH = 'results'
DEFAULT_CONFIG.ENVIRONMENT.MLFLOW_BASE_PATH = 'mlflow'
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
DEFAULT_CONFIG.MODEL = CN()
DEFAULT_CONFIG.MODEL.TYPE = ''
DEFAULT_CONFIG.MODEL.SPEC_NAME = ''
DEFAULT_CONFIG.MODEL.ABBR = CN(visible=False)
# DEFAULT_CONFIG.MODEL.ABBR.model_name = 'model_abbreviation'
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
DEFAULT_CONFIG.TRAIN = CN()
# _C.TRAIN.BASE_LR = 5e-4
DEFAULT_CONFIG.TRAIN.MONITOR = 'valid/epoch/loss'
DEFAULT_CONFIG.TRAIN.MONITOR_MODE = 'min'
# Trainer settings
DEFAULT_CONFIG.TRAIN.TRAINER = CN()
DEFAULT_CONFIG.TRAIN.TRAINER.accelerator = 'gpu'
DEFAULT_CONFIG.TRAIN.TRAINER.precision = '16-mixed'
DEFAULT_CONFIG.TRAIN.TRAINER.strategy = 'auto'
DEFAULT_CONFIG.TRAIN.TRAINER.min_epochs = 1
DEFAULT_CONFIG.TRAIN.TRAINER.max_epochs = 50
DEFAULT_CONFIG.TRAIN.TRAINER.overfit_batches = 0

DEFAULT_CONFIG.TRAIN.TRAINER.set_typecheck_exclude_keys(['precision', 'overfit_batches'])
DEFAULT_CONFIG.TRAIN.TRAINER.set_invisible_keys(['accelerator'])

# Callbacks
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
DEFAULT_CONFIG.TRAIN.USE_CHECKPOINT = False
DEFAULT_CONFIG.TRAIN.CHECKPOINT_TOPK = 1
DEFAULT_CONFIG.TRAIN.CHECKPOINT_SAVELAST = False
# Whether to use earlystopping
# could be overwritten by command line argument
DEFAULT_CONFIG.TRAIN.USE_EARLYSTOPPING = False
DEFAULT_CONFIG.TRAIN.EARLYSTOPPING_PATIENCE = 10

DEFAULT_CONFIG.TRAIN.LOG_LOSS = True
DEFAULT_CONFIG.TRAIN.LOG_LEARNINGRATE = True
# -----------------------------------------------------------------------------
# Logger settings
# -----------------------------------------------------------------------------
DEFAULT_CONFIG.LOGGER = CN()
DEFAULT_CONFIG.LOGGER.LOG_MODEL = True
# Logger to be used
DEFAULT_CONFIG.LOGGER.NAME = 'MLFlowLogger'
DEFAULT_CONFIG.LOGGER.MLFLOW = CN(visible=False)
DEFAULT_CONFIG.LOGGER.MLFLOW.tracking_uri = 'http://www.leike.xyz:5000'
DEFAULT_CONFIG.LOGGER.MLFLOW.artifact_location = 'mlflow'
# Experiment name to be logged
# could be overwritten by command line argument
DEFAULT_CONFIG.LOGGER.EXPERIMENT_NAME = 'Default'

DEFAULT_CONFIG.LOGGER.set_invisible_keys(['NAME', 'LOG_MODEL'])

# Experiment tag to be logged
DEFAULT_CONFIG.LOGGER.TAGS = CN(visible=False)
DEFAULT_CONFIG.LOGGER.TAGS.TASK = 'SlagSegmentation'
DEFAULT_CONFIG.LOGGER.TAGS.TYPE = 'test'
# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
DEFAULT_CONFIG.PREDICT = CN()
DEFAULT_CONFIG.PREDICT.RESULT_PATH = 'results'
# Path to output results, could be overwritten by command line argument
DEFAULT_CONFIG.PREDICT.WRITER = CN(visible=False)
DEFAULT_CONFIG.PREDICT.WRITER.image_suffixs = ''
DEFAULT_CONFIG.PREDICT.WRITER.image_format = 'png'
DEFAULT_CONFIG.PREDICT.WRITER.concatenate = False
DEFAULT_CONFIG.PREDICT.WRITER.log_prediction = False
DEFAULT_CONFIG.PREDICT.WRITER.log_folder = ''

DEFAULT_CONFIG.PREDICT.WRITER.set_typecheck_exclude_keys(['concatenate'])
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Path to output folder, overwritten by command line argument
DEFAULT_CONFIG.CONFIG_OUTPUT_PATH = 'configs'
DEFAULT_CONFIG.TAG = 'Default'
DEFAULT_CONFIG.FULL_DUMP = False

DEFAULT_CONFIG.set_invisible_keys(['FULL_DUMP'])