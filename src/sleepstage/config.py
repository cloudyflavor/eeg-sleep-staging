"""설정 로딩과 해시. 실험 하나 = 설정 파일 하나. 해시는 산출물 추적과 멱등 재개의 축."""

from __future__ import annotations

import copy
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
    """설정·출처·해시 묶음 — 항상 같이 다니게 한다."""

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
        """ "a.b=값" 목록을 적용한 새 Config. 없는 키는 예외 — 오타가 새 실험이 되는 것 방지."""
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
