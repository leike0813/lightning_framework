from collections.abc import Sequence
from lightning.pytorch.callbacks import Callback
from .metric_hook import MetricHook
from .metric_callback import MetricCallback
from .roc_callback import ROCCallback
import lightning as L


class DummyCallback(Callback):
    def __init__(self):
        pass


def build_metric_callback(node):
    node.defrost()
    for param, value in node.GLOBAL_PARAMS.items():
        for stage in ['TRAINING', 'VALIDATION', 'TEST']:
            for metric_type, metric_node in node[stage].items():
                valid_global_params = metric_node.get('VALID_GLOBAL_PARAMS', None)
                if valid_global_params is not None:
                    metric_node.pop('VALID_GLOBAL_PARAMS')
                    if param in valid_global_params:
                        metric_node[param] = value
                else:
                    metric_node[param] = value
    node.freeze()

    return MetricCallback(
        training_metric_dict=node.get('TRAINING', {}), validation_metric_dict=node.get('VALIDATION', {}),
        test_metric_dict=node.get('TEST', {}), category=node.get('CATEGORY', 'default'),
        append_category_name=node.get('APPEND_CATEGORY_NAME', False)
    )


def build_roc_callback(node):
    if 'ROC' not in node.keys():
        return DummyCallback()

    node.defrost()
    for param, value in node.GLOBAL_PARAMS.items():
        node['ROC'][param] = value
    node.freeze()

    return ROCCallback(
        category=node.get('CATEGORY', 'default'),
        append_category_name=node.get('APPEND_CATEGORY_NAME', False),
        **node['ROC']
    )


def install_metric_hook(lightningModule):
    if not issubclass(lightningModule, L.LightningModule):
        raise TypeError('Can only be used to lightning.LightningModule')

    class _LightningModule(lightningModule, MetricHook):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    _LightningModule.__name__ = lightningModule.__name__

    return _LightningModule
