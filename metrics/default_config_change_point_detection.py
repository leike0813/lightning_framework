from pytorch_framework.utils import CustomCfgNode as CN


DEFAULT_CONFIG = CN(visible=False)

DEFAULT_CONFIG.CATEGORY = 'default'
DEFAULT_CONFIG.APPEND_CATEGORY_NAME = False

DEFAULT_CONFIG.GLOBAL_PARAMS = CN(visible=False)
DEFAULT_CONFIG.GLOBAL_PARAMS.padding_logits_value = -100

DEFAULT_CONFIG.TRAINING = CN(visible=False)

DEFAULT_CONFIG.TRAINING.AbsolutePositionError = CN(visible=False)
DEFAULT_CONFIG.TRAINING.AbsolutePositionError.LEVEL = 3
DEFAULT_CONFIG.TRAINING.AbsolutePositionError.ALIAS = 'APE'
DEFAULT_CONFIG.TRAINING.RelativePositionError = CN(visible=False)
DEFAULT_CONFIG.TRAINING.RelativePositionError.LEVEL = 3
DEFAULT_CONFIG.TRAINING.RelativePositionError.ALIAS = 'RPE'

DEFAULT_CONFIG.VALIDATION = CN(visible=False)

DEFAULT_CONFIG.VALIDATION.AbsolutePositionError = CN(visible=False)
DEFAULT_CONFIG.VALIDATION.AbsolutePositionError.LEVEL = 2
DEFAULT_CONFIG.VALIDATION.AbsolutePositionError.ALIAS = 'APE'
DEFAULT_CONFIG.VALIDATION.RelativePositionError = CN(visible=False)
DEFAULT_CONFIG.VALIDATION.RelativePositionError.LEVEL = 2
DEFAULT_CONFIG.VALIDATION.RelativePositionError.ALIAS = 'RPE'

DEFAULT_CONFIG.TEST = CN(visible=False)

DEFAULT_CONFIG.TEST.AbsolutePositionError = CN(visible=False)
DEFAULT_CONFIG.TEST.AbsolutePositionError.LEVEL = 2
DEFAULT_CONFIG.TEST.AbsolutePositionError.ALIAS = 'APE'
DEFAULT_CONFIG.TEST.RelativePositionError = CN(visible=False)
DEFAULT_CONFIG.TEST.RelativePositionError.LEVEL = 2
DEFAULT_CONFIG.TEST.RelativePositionError.ALIAS = 'RPE'
DEFAULT_CONFIG.TEST.EarlyDetectionRate = CN(visible=False)
DEFAULT_CONFIG.TEST.EarlyDetectionRate.LEVEL = 2
DEFAULT_CONFIG.TEST.EarlyDetectionRate.ALIAS = 'EDR'
DEFAULT_CONFIG.TEST.EarlyDetectionRate.tolerance = 0
DEFAULT_CONFIG.TEST.LateDetectionRate = CN(visible=False)
DEFAULT_CONFIG.TEST.LateDetectionRate.LEVEL = 2
DEFAULT_CONFIG.TEST.LateDetectionRate.ALIAS = 'LDR'
DEFAULT_CONFIG.TEST.LateDetectionRate.tolerance = 0
