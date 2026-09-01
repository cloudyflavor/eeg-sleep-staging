"""설정 파일을 읽고 내용을 짧은 문자열로 요약한다.

값을 코드에 적지 않고 파일로 빼두면 코드를 고치지 않고 실험을 바꿀 수 있다.
설정 내용을 요약한 문자열은 결과 파일 이름이 되어, 어떤 설정으로 만든 결과인지
나중에 찾아볼 수 있게 한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def config_hash(cfg: dict[str, Any], length: int = 12) -> str:
    """설정 내용을 짧은 문자열로 요약한다. 항목 순서가 달라도 내용이 같으면 같은 값이 나온다."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class Config:
    """설정 내용과 파일 경로와 요약 문자열을 함께 들고 다니는 묶음.

    따로 넘기면 어느 순간 짝이 어긋나 잘못된 이름으로 저장될 수 있다.
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

    def override(self, pairs: list[str]) -> Config:
        """일부 값만 바꾼 새 설정을 만든다.

        ``"features.filter=none"`` 처럼 적는다. 없는 항목을 지정하면 오류를 낸다.
        오타를 그냥 두면 의도하지 않은 설정으로 실험이 돌아간다.
        """
        if not pairs:
            return self
        data = copy.deepcopy(self.data)
        for pair in pairs:
            dotted, _, raw = pair.partition("=")
            if not _:
                raise ValueError(f"'키=값' 형식이어야 합니다: {pair!r}")
            *parents, leaf = dotted.split(".")
            node = data
            for part in parents:
                if part not in node:
                    raise KeyError(f"설정에 '{dotted}' 가 없습니다 (출처: {self.path})")
                node = node[part]
            if leaf not in node:
                raise KeyError(f"설정에 '{dotted}' 가 없습니다 (출처: {self.path})")
            node[leaf] = yaml.safe_load(raw)
        return Config(self.path, data, config_hash(data))
