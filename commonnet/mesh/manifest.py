"""The network manifest: which model the peers serve and where its stage boundaries fall.

Every peer serves a window that starts and ends on a boundary, so windows are unions of
consecutive stages and two peers with the same window run exactly the same layers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Manifest:
    id: str
    version: int
    model: dict
    boundaries: tuple[int, ...]
    gib_per_layer: dict
    fixed_gib: float = 0.0
    min_replicas: int = 2
    seeds: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, source) -> "Manifest":
        """From a dict, a JSON string, or a path to a JSON file."""
        if isinstance(source, Manifest):
            return source
        if isinstance(source, (str, Path)):
            text = str(source)
            if text.lstrip().startswith("{"):
                data = json.loads(text)
            else:
                try:
                    data = json.loads(Path(text).read_text())
                except OSError as e:
                    raise ManifestError(f"cannot read manifest {text}: {e}") from None
                except ValueError:
                    raise ManifestError(f"{text} is not JSON") from None
        elif isinstance(source, dict):
            data = source
        else:
            raise ManifestError(f"cannot load a manifest from {type(source).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        if not isinstance(data, dict):
            raise ManifestError("manifest must be a JSON object")
        try:
            manifest = cls(
                id=data["id"], version=int(data.get("version", 1)), model=dict(data["model"]),
                boundaries=tuple(data["boundaries"]), gib_per_layer=dict(data.get("gib_per_layer") or {}),
                fixed_gib=float(data.get("fixed_gib", 0.0)), min_replicas=int(data.get("min_replicas", 2)),
                seeds=tuple(data.get("seeds") or ()))
        except (KeyError, TypeError, ValueError) as e:
            raise ManifestError(f"manifest is missing or has a bad field: {e!r}") from None
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id or len(self.id) > 64:
            raise ManifestError("manifest id must be a non-empty string of at most 64 characters")
        layers = self.model.get("num_layers")
        if type(layers) is not int or layers < 1:
            raise ManifestError("model.num_layers must be a positive integer")
        b = self.boundaries
        if len(b) < 2 or any(type(x) is not int for x in b):
            raise ManifestError("boundaries must be a list of at least two integers")
        if b[0] != 0 or b[-1] != layers or any(y <= x for x, y in zip(b, b[1:])):
            raise ManifestError(f"boundaries must rise strictly from 0 to num_layers ({layers})")
        if self.min_replicas < 1:
            raise ManifestError("min_replicas must be at least 1")
        if not isinstance(self.model.get("repos", {}), dict):
            raise ManifestError("model.repos must be an object")

    @property
    def num_layers(self) -> int:
        return self.model["num_layers"]

    def stages(self) -> list[tuple[int, int]]:
        return list(zip(self.boundaries, self.boundaries[1:]))

    def repo(self, backend: str) -> str | None:
        return (self.model.get("repos") or {}).get(backend)

    @property
    def client_repo(self) -> str | None:
        return self.model.get("client_repo")

    def per_layer(self, backend: str) -> float:
        """GiB per layer for `backend`; a backend the manifest does not list is costed at the largest listed."""
        if backend in self.gib_per_layer:
            return float(self.gib_per_layer[backend])
        return max((float(v) for v in self.gib_per_layer.values()), default=0.0)

    def window_gib(self, a: int, b: int, backend: str) -> float:
        return self.fixed_gib + (b - a) * self.per_layer(backend)

    def is_window(self, a: int, b: int) -> bool:
        return a in self.boundaries and b in self.boundaries and a < b

    def windows_fitting(self, gib: float, backend: str) -> list[tuple[int, int]]:
        """Every boundary-aligned window whose weights fit in `gib`, longest first."""
        out = [(a, b) for i, a in enumerate(self.boundaries) for b in self.boundaries[i + 1:]
               if self.window_gib(a, b, backend) <= gib]
        out.sort(key=lambda w: (-(w[1] - w[0]), w[0]))
        return out

    def to_dict(self) -> dict:
        return {"id": self.id, "version": self.version, "model": self.model,
                "boundaries": list(self.boundaries), "gib_per_layer": self.gib_per_layer,
                "fixed_gib": self.fixed_gib, "min_replicas": self.min_replicas, "seeds": list(self.seeds)}
