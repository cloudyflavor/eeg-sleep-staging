"""앞뒤 흐름까지 보고 판단하는 신경망.

조각마다 encoder 로 특징을 뽑은 뒤, 그 특징들을 순서대로 훑어 판단한다.
앞부분이 조각 하나만 보는 신경망과 완전히 같아서, 두 모델의 차이가 흐름을 보느냐
마느냐 하나로 좁혀진다.

양방향이라 뒤쪽 조각도 참고한다. 그만큼 판정이 늦어지고, 얼마나 늦어지는지는
묶음 길이가 정한다.
"""

import torch
from torch import nn

from sleepstage.experiment.baselines.deep.encoder import SleepCNN


class SleepCNNLSTM(nn.Module):
    """조각마다 특징을 뽑은 뒤 앞뒤 흐름까지 보고 판단한다.

    앞부분은 조각 하나만 보는 신경망과 같다. 그 결과를 순서대로 이어 붙여 흐름을
    보는 층에 넣는다. 양방향이라 뒤쪽 조각도 참고하므로 그만큼 판정이 늦어진다.
    """

    def __init__(self, n_channels: int = 2, n_classes: int = 4, hidden: int = 128):
        super().__init__()
        self.encoder = SleepCNN(n_channels=n_channels, n_classes=n_classes)
        self.encoder.head = nn.Identity()  # 조각별 판정 대신 특징만 꺼낸다
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(묶음, 조각, 채널, 샘플) 을 받아 조각마다 판정을 낸다."""
        batch, steps = x.shape[0], x.shape[1]
        flat = x.reshape(batch * steps, *x.shape[2:])
        features = self.encoder(flat).reshape(batch, steps, -1)
        out, _ = self.rnn(features)
        return self.head(self.dropout(out))
