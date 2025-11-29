from enum import IntEnum, IntFlag
import warnings
from torch import nn
import torchmetrics as TM
from . import custom_metrics as CM
from lightning.pytorch.callbacks import Callback

TM_DOMAINS = {}
try:
    import torchmetrics.aggregation as TM_aggregation
    TM_DOMAINS['aggregation'] = TM_aggregation
except ImportError:
    pass
try:
    import torchmetrics.audio as TM_audio
    TM_DOMAINS['audio'] = TM_audio
except ImportError:
    pass
try:
    import torchmetrics.classification as TM_classification
    TM_DOMAINS['classification'] = TM_classification
except ImportError:
    pass
try:
    import torchmetrics.clustering as TM_clustering
    TM_DOMAINS['clustering'] = TM_clustering
except ImportError:
    pass
try:
    import torchmetrics.detection as TM_detection
    TM_DOMAINS['detection'] = TM_detection
except ImportError:
    pass
try:
    import torchmetrics.image as TM_image
    TM_DOMAINS['image'] = TM_image
except ImportError:
    pass
try:
    import torchmetrics.nominal as TM_nominal
    TM_DOMAINS['nominal'] = TM_nominal
except ImportError:
    pass
try:
    import torchmetrics.retrieval as TM_retrieval
    TM_DOMAINS['retrieval'] = TM_retrieval
except ImportError:
    pass
try:
    import torchmetrics.segmentation as TM_segmentation
    TM_DOMAINS['segmentation'] = TM_segmentation
except ImportError:
    pass
try:
    import torchmetrics.shape as TM_shape
    TM_DOMAINS['shape'] = TM_shape
except ImportError:
    pass
try:
    import torchmetrics.text as TM_text
    TM_DOMAINS['text'] = TM_text
except ImportError:
    pass


def _get_TM_module(domain):
    if domain is None or domain == '':
        return TM
    domain_parts = domain.split('.')
    parent = TM
    for subdomain in domain_parts:
        parent = getattr(parent, subdomain, parent)

    return parent


class EvaluationLevel(IntFlag):
    BATCH = 1
    EPOCH = 2


class MetricCallback(Callback):
    stage_conversion = {'fit': 'train', 'validate': 'valid', 'test': 'test'}

    def __init__(self, training_metric_dict=None, validation_metric_dict=None, test_metric_dict=None, category='defualt',
                 append_category_name=False):
        super().__init__()
        self.training_metric_dict = training_metric_dict
        self.validation_metric_dict = validation_metric_dict
        self.test_metric_dict = test_metric_dict
        self.category = category
        self.append_category_name = append_category_name
        self.__TRAIN_METRICS_MOVED = False
        self.__VALID_METRICS_MOVED = False
        self.__TEST_METRICS_MOVED = False

    @property
    def state_key(self):
        return self.category

    def state_dict(self):
        return {
            'training_metric_dict': self.training_metric_dict,
            'validation_metric_dict': self.validation_metric_dict,
            'test_metric_dict': self.test_metric_dict,
            'category': self.category,
            'append_category_name': self.append_category_name,
        }


    def setup(self, trainer, pl_module, stage):
        if stage == 'fit':
            self.training_batch_metrics = nn.ModuleDict()
            self.training_epoch_metrics = nn.ModuleDict()
            self.setup_metrics(self.training_batch_metrics, self.training_epoch_metrics,
                               self.training_metric_dict, self.stage_conversion[stage])

            self.validation_batch_metrics = nn.ModuleDict()
            self.validation_epoch_metrics = nn.ModuleDict()
            self.setup_metrics(self.validation_batch_metrics, self.validation_epoch_metrics,
                               self.validation_metric_dict, 'valid')
        if stage == 'validate':
            if not hasattr(self, 'validation_batch_metrics'):
                self.validation_batch_metrics = nn.ModuleDict()
                self.validation_epoch_metrics = nn.ModuleDict()
                self.setup_metrics(self.validation_batch_metrics, self.validation_epoch_metrics,
                                   self.validation_metric_dict, self.stage_conversion[stage])
        if stage == 'test':
            self.test_batch_metrics = nn.ModuleDict()
            self.test_epoch_metrics = nn.ModuleDict()
            self.setup_metrics(self.test_batch_metrics, self.test_epoch_metrics,
                               self.test_metric_dict, self.stage_conversion[stage])

    def on_train_start(self, trainer, pl_module):
        self._data_fetched = False
        if not self.__TRAIN_METRICS_MOVED:
            for metric_func in self.training_batch_metrics.values():
                metric_func.to(pl_module.device)
            for metric_func in self.training_epoch_metrics.values():
                metric_func.to(pl_module.device)

    def on_validation_start(self, trainer, pl_module):
        self._data_fetched = False
        if not self.__VALID_METRICS_MOVED:
            for metric_func in self.validation_batch_metrics.values():
                metric_func.to(pl_module.device)
            for metric_func in self.validation_epoch_metrics.values():
                metric_func.to(pl_module.device)

    def on_test_start(self, trainer, pl_module):
        self._data_fetched = False
        if not self.__TEST_METRICS_MOVED:
            for metric_func in self.test_batch_metrics.values():
                metric_func.to(pl_module.device)
            for metric_func in self.test_epoch_metrics.values():
                metric_func.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.calc_batch_metrics(outputs, self.training_batch_metrics, self.training_epoch_metrics, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self.calc_epoch_metrics(self.training_epoch_metrics, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.calc_batch_metrics(outputs, self.validation_batch_metrics, self.validation_epoch_metrics, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.calc_epoch_metrics(self.validation_epoch_metrics, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.calc_batch_metrics(outputs, self.test_batch_metrics, self.test_epoch_metrics, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self.calc_epoch_metrics(self.test_epoch_metrics, pl_module)

    def setup_metrics(self, metrics_batch_container, metrics_epoch_container, metric_dict, stage):
        containers = {'batch': metrics_batch_container, 'epoch': metrics_epoch_container}
        for metric_type, params in metric_dict.items():
            level = EvaluationLevel(params.pop('LEVEL', 2))
            alias = params.pop('ALIAS', None)
            if not isinstance(level, (int, EvaluationLevel)) or level < 1 or level > 3:
                raise ValueError('level must be valid metrics.EvaluationLevel flag or integer')
            _levels = []
            if EvaluationLevel.BATCH in level:
                _levels.append('batch')
            if EvaluationLevel.EPOCH in level:
                _levels.append('epoch')

            domain = params.pop('DOMAIN', None)
            tm_module = _get_TM_module(domain)
            metric_generator = getattr(
                CM,
                metric_type,
                getattr(
                    tm_module,
                    metric_type,
                    getattr(
                        TM_DOMAINS.get(domain, TM),
                        metric_type,
                        None
                    )
                )
            )
            if metric_generator is None:
                raise ValueError('metric type {mt} not found'.format(mt=metric_type))
            metric_func = metric_generator(**params)

            for lvl in _levels:
                reg_name = '{stg}/{lvl}/{nme}{cat}'.format(
                    stg=stage,
                    lvl=lvl,
                    nme=alias if alias else metric_type,
                    cat='_' + self.category if self.append_category_name else ''
                )
                containers[lvl][reg_name] = metric_func

    def fetch_data_for_calc(self, outputs):
        keys = ['metric_{cat}'.format(cat=self.category), 'metrics_{cat}'.format(cat=self.category), 'metric_src_{cat}'.format(cat=self.category), self.category]
        if self.category == 'default':
            keys.extend(['metric', 'metrics', 'metric_src'])
        for key in keys:
            data = outputs.get(key)
            if data is not None:
                return data
        return None

    def calc_batch_metrics(self, outputs, metrics_batch_container, metrics_epoch_container, pl_module):
        # input = self.collate_fn(input)
        input = self.fetch_data_for_calc(outputs)
        if input is not None:
            self._data_fetched = True
            for metric_name, metric_fn in metrics_batch_container.items():
                metric_value = metric_fn(*input)
                self.log(metric_name, metric_value)
            for _, metric_fn in metrics_epoch_container.items():
                if metric_fn not in metrics_batch_container.values():
                    metric_fn(*input)

    def calc_epoch_metrics(self, metrics_epoch_container, pl_module):
        if self._data_fetched:
            for metric_name, metric_fn in metrics_epoch_container.items():
                metric_value = metric_fn.compute()
                self.log(metric_name, metric_value)
                metric_fn.reset()
            self._data_fetched = False
