from typing import Any, Callable, List, Optional, Tuple, Type, Union, Sequence, Iterator

import torch
from torchmetrics.metric import Metric
from torch import Tensor


class BasePositionMetric(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
            self,
            padding_logits_value = -100,
            validate_args: bool = True,
            **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if validate_args:
            assert padding_logits_value < 0, "padding_logits_value must be < 0"
            if padding_logits_value > -100:
                print("WARNING: padding_logits_value may be too high to distinguish between padding and model outputs. Consider using a value < -100.")

        self.validate_args = validate_args
        self.padding_logits_value = padding_logits_value

        self.add_state('pe', [])
        self.add_state('length', [])

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.validate_args:
            assert preds.dim() == 2, "preds must be 2-dim tensor"
            assert target.dim() == 1, "target must be 1-dim tensor"
            assert target.max() < preds.shape[-1], "target can not be greater than the sequence length"
            if not ((preds > 1).any() or (preds < 0).any()):
                print("WARNING: preds seems to be probabilities, thus can not infer sequence length from preds. Consider using raw logits as preds with large negative value as padding flag.")

        lengths = (preds > self.padding_logits_value).sum(dim=-1)
        self.length += lengths.tolist()
        pred_positions = torch.argmax(preds, dim=-1)
        position_errors = pred_positions - target
        self.pe += position_errors.tolist()


class AbsolutePositionError(BasePositionMetric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def compute(self) -> Tensor:
        return torch.tensor(sum([abs(pe) for pe in self.pe]) / len(self.pe))


class RelativePositionError(BasePositionMetric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def compute(self) -> Tensor:
        return torch.tensor(sum([abs(pe) / _len for pe, _len in zip(self.pe, self.length)]) / len(self.pe))


class EarlyDetectionRate(BasePositionMetric):
    def __init__(
            self,
            tolerance: int,
            padding_logits_value: Union[int, float] = -100,
            validate_args: bool = True,
            **kwargs: Any,
    ):
        super().__init__(padding_logits_value, validate_args, **kwargs)
        if validate_args:
            assert isinstance(tolerance, int), 'tolerance must be an integer.'
            assert tolerance >= 0, 'tolerance must be positive.'

        self.tolerance = tolerance

    def compute(self) -> Tensor:
        return torch.tensor(len([pe for pe in self.pe if pe < -self.tolerance]) / len(self.pe))


class LateDetectionRate(BasePositionMetric):
    def __init__(
            self,
            tolerance,
            padding_logits_value=-100,
            validate_args: bool = True,
            **kwargs: Any,
    ):
        super().__init__(padding_logits_value, validate_args, **kwargs)
        if validate_args:
            assert tolerance >= 0, 'tolerance must be greater than 0.'

        self.tolerance = tolerance

    def compute(self) -> Tensor:
        return torch.tensor(len([pe for pe in self.pe if pe > self.tolerance]) / len(self.pe))


if __name__ == '__main__':
    def test_position_metrics():
        print("开始测试位置度量指标...")

        # 创建测试数据
        # 注意：preds必须是2维张量，target必须是1维张量

        # 案例1：完全匹配
        # 创建一个批次大小为1的预测张量，序列长度为5
        # 在位置1、2、3处有高logits值，其他位置为padding值
        preds1 = torch.tensor([[-1.5, 10, -2.7, -4.8, -7.3]])  # 预测位置为1
        target1 = torch.tensor([1])  # 目标位置为1

        # 案例2：预测位置早于目标位置（早期检测）
        preds2 = torch.tensor([[-1.5, 10, -2.7, -4.8, -7.3]])  # 预测位置为1
        target2 = torch.tensor([2])  # 目标位置为2

        # 案例3：预测位置晚于目标位置（晚期检测）
        preds3 = torch.tensor([[-1.5, -2.7, 10, -4.8, -7.3]])  # 预测位置为2
        target3 = torch.tensor([1])  # 目标位置为1

        # 案例4：多个样本的批次
        preds4 = torch.tensor([
            [-1.5, 10, -2.7, -4.8, -100],  # 预测位置为1，序列长度为4
            [-1.5, -2.7, 10, -100, -100],  # 预测位置为2，序列长度为3
            [-1.5, -2.7, -4.8, 10, -7.3]   # 预测位置为3，序列长度为5
        ])
        target4 = torch.tensor([1, 2, 4])  # 目标位置分别为1, 2, 4

        # 创建度量指标实例
        ape = AbsolutePositionError()
        rpe = RelativePositionError()
        edr = EarlyDetectionRate(tolerance=0)
        ldr = LateDetectionRate(tolerance=0)

        # 测试案例1：完全匹配
        print("\n=== 测试案例1：完全匹配 ===")
        ape.update(preds1, target1)
        rpe.update(preds1, target1)
        edr.update(preds1, target1)
        ldr.update(preds1, target1)

        print(f"绝对位置误差 (APE): {ape.compute().item():.4f}")
        print(f"相对位置误差 (RPE): {rpe.compute().item():.4f}")
        print(f"早期检测率 (EDR): {edr.compute().item():.4f}")
        print(f"晚期检测率 (LDR): {ldr.compute().item():.4f}")

        # 重置度量指标
        ape.reset()
        rpe.reset()
        edr.reset()
        ldr.reset()

        # 测试案例2：早期检测
        print("\n=== 测试案例2：早期检测 ===")
        ape.update(preds2, target2)
        rpe.update(preds2, target2)
        edr.update(preds2, target2)
        ldr.update(preds2, target2)

        print(f"绝对位置误差 (APE): {ape.compute().item():.4f}")
        print(f"相对位置误差 (RPE): {rpe.compute().item():.4f}")
        print(f"早期检测率 (EDR): {edr.compute().item():.4f}")
        print(f"晚期检测率 (LDR): {ldr.compute().item():.4f}")

        # 重置度量指标
        ape.reset()
        rpe.reset()
        edr.reset()
        ldr.reset()

        # 测试案例3：晚期检测
        print("\n=== 测试案例3：晚期检测 ===")
        ape.update(preds3, target3)
        rpe.update(preds3, target3)
        edr.update(preds3, target3)
        ldr.update(preds3, target3)

        print(f"绝对位置误差 (APE): {ape.compute().item():.4f}")
        print(f"相对位置误差 (RPE): {rpe.compute().item():.4f}")
        print(f"早期检测率 (EDR): {edr.compute().item():.4f}")
        print(f"晚期检测率 (LDR): {ldr.compute().item():.4f}")

        # 重置度量指标
        ape.reset()
        rpe.reset()
        edr.reset()
        ldr.reset()

        # 测试案例4：多个样本的批次
        print("\n=== 测试案例4：多个样本的批次 ===")
        ape.update(preds4, target4)
        rpe.update(preds4, target4)
        edr.update(preds4, target4)
        ldr.update(preds4, target4)

        print(f"绝对位置误差 (APE): {ape.compute().item():.4f}")
        print(f"相对位置误差 (RPE): {rpe.compute().item():.4f}")
        print(f"早期检测率 (EDR): {edr.compute().item():.4f}")
        print(f"晚期检测率 (LDR): {ldr.compute().item():.4f}")

        print("\n所有测试完成！")

    # 运行测试
    try:
        test_position_metrics()
    except Exception as e:
        print(f"\n测试过程中出现错误：{str(e)}")
