"""Central resource resolver and future encrypted-resource provider hook."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from common.paths import SDK_ROOT


ResourceProvider = Callable[[str], Path | str]
_provider: ResourceProvider | None = None


def register_resource_provider(provider: ResourceProvider | None) -> None:
    """Install a resolver used by a future encrypted resource container."""
    global _provider
    _provider = provider


def resource_path(relative: str | os.PathLike) -> Path:
    normalized = Path(relative)
    if normalized.is_absolute():
        if normalized.exists():
            return normalized
        raise FileNotFoundError(f"resource does not exist: {normalized}")
    key = normalized.as_posix()
    if _provider is not None:
        supplied = Path(_provider(key)).expanduser().resolve()
        if not supplied.exists():
            raise FileNotFoundError(
                f"resource provider returned a missing path for {key}: {supplied}")
        return supplied
    roots = []
    configured = os.environ.get("STOUCH_GLOVE_RESOURCE_ROOT")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend((SDK_ROOT / "assets", Path.cwd() / "assets"))
    for root in roots:
        candidate = (root / normalized).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"resource {key!r} was not found below: "
        + ", ".join(str(path.resolve()) for path in roots))


def hand_assets_dir() -> Path:
    return resource_path("hand")


def resolve_hand_assets_dir(value: str | os.PathLike = "assets/hand") -> Path:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if candidate.as_posix().rstrip("/") in {"assets/hand", "hand"}:
        return hand_assets_dir()
    return resource_path(candidate)


def hand_geometry_path(filename: str = "hand_measured_runtime_v1.json") -> Path:
    return resource_path(Path("hand_geometry") / filename)

