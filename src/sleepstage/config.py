"""설정 로딩과 해시.

실험 하나 = 설정 파일 하나. 코드에 값을 하드코딩하지 않는다.

설정 해시를 산출물에 함께 저장하면 "이 파일이 어떤 설정으로 만들어졌는지" 를
나중에 추적할 수 있고, 재실행 시 설정이 바뀌었는지 판단해 건너뛸 수 있다
(DeepSleepNet/TinySleepNet 은 시작할 때 출력 디렉터리를 통째로 지운다).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def config_hash(cfg: dict[str, Any], length: int = 12) -> str:
    """설정을 정규화해 짧은 해시로. 키 순서가 달라도 같은 해시가 나오게 정렬한다."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class Config:
    """설정 딕셔너리에 출처와 해시를 붙인 것.

    ``cfg["epoch"]["seconds"]`` 처럼 딕셔너리로 그대로 쓴다. 얇은 껍데기일 뿐이고,
    존재 이유는 **설정과 그 해시가 항상 같이 다니게** 하는 것이다.
    둘을 따로 넘기면 어느 순간 짝이 어긋난 채 저장된다.
    """

    path: Path
    data: dict[str, Any]
    hash: str

    @classmethod
    def load(cls, path: str | Path) -> Config:
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(p, data, config_hash(data))

    def __getitem__(self, key: str) -> Any:
        return self.data[key]
