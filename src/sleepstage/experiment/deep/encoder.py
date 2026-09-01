"""조각 하나에서 특징을 뽑는 신경망.

짧은 구간을 보는 갈래와 긴 구간을 보는 갈래를 나란히 둔다. 짧은 쪽은 순간적으로
나타나는 파형을, 긴 쪽은 느리게 이어지는 리듬을 잡는다.

이 신경망만으로도 조각 하나를 보고 수면 단계를 판단할 수 있다. 앞뒤 흐름까지
보려면 sequence.py 가 이것을 특징 추출기로 쓴다.
"""

import torch
from torch import nn


class SleepCNN(nn.Module):
    def __init__(self, n_channels: int = 2, n_classes: int = 4, sfreq: int = 100):
        super().__init__()
        self.small = self._branch(n_channels, kernel=sfreq // 2, stride=sfreq // 16, pool=8)
        self.large = self._branch(n_channels, kernel=sfreq * 4, stride=sfreq // 2, pool=4)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.LazyLinear(n_classes)

    @staticmethod
    def _branch(n_channels: int, kernel: int, stride: int, pool: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel, stride, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(pool),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, 8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        both = torch.cat([self.small(x).flatten(1), self.large(x).flatten(1)], dim=1)
        return self.head(self.dropout(both))
