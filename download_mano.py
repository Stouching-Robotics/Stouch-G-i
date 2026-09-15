#!/usr/bin/env python3
"""下载/安装 MANO 手部模型，并重建本地 HAND+2mm 运行时资源。

MANO 模型（Max-Planck Institute）仅授权非商业科研/教育用途，且**禁止再分发**，
因此本仓库不包含 MANO 的模型文件。每位使用者需自行到官网注册下载，再用本脚本
把文件装入 `assets/hand/`。

用法
----
    python download_mano.py                              # 查看当前安装状态
    python download_mano.py --open                       # 打开 MANO 下载页面
    python download_mano.py --install MANO_LEFT.pkl MANO_RIGHT.pkl
    python download_mano.py --install <解压后的目录>

`--install` 会把两个 PKL 复制到 `assets/hand/models/`，并据此重建
`assets/hand/HAND_*_PINKY_PLUS_2MM.npz`（求解和 3D 显示实际加载的是后者）。
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import webbrowser
from pathlib import Path

import numpy as np

try:
    import scipy.sparse as _sp  # noqa: F401  仅用于判断稀疏矩阵
    import chumpy  # noqa: F401  MANO pkl 依赖 chumpy 反序列化
except ImportError as exc:  # pragma: no cover
    print(f"缺少依赖：{exc}。请先按 README 安装 requirements.txt。", file=sys.stderr)
    sys.exit(1)

BUNDLE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = BUNDLE_ROOT / "assets" / "hand" / "models"
HAND_DIR = BUNDLE_ROOT / "assets" / "hand"

MANO_DOWNLOAD_URL = "https://mano.is.tue.mpg.de/"

# 与运行时一致：HAND+2mm 只是把 MANO 的模板数据原样打包，外加运行期参数。
PINKY_EXTRA_LENGTH_M = 0.002
LOCAL_DERIVATIVE_KIND = "pinky_plus_2mm"


def _dense(value) -> np.ndarray:
    """把 chumpy / 稀疏矩阵统一转成稠密 numpy 数组。"""
    if hasattr(value, "r"):  # chumpy Ch 对象
        value = value.r
    if hasattr(value, "toarray"):  # scipy 稀疏矩阵
        value = value.toarray()
    return np.asarray(value)


def build_npz(pkl_path: Path, npz_path: Path) -> None:
    """从一个 MANO pkl 重建 HAND+2mm 的 npz 运行时资源。"""
    with pkl_path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")

    payload = {
        "v_template": _dense(model["v_template"]).astype(np.float64),
        "shapedirs": _dense(model["shapedirs"]).astype(np.float64),
        "posedirs": _dense(model["posedirs"]).astype(np.float64),
        "J_regressor": _dense(model["J_regressor"]).astype(np.float64),
        "weights": _dense(model["weights"]).astype(np.float64),
        "kintree_table": _dense(model["kintree_table"]).astype(np.int64),
        "pinky_extra_length_m": np.float64(PINKY_EXTRA_LENGTH_M),
        "local_derivative_kind": np.str_(LOCAL_DERIVATIVE_KIND),
        "local_demo_only": np.bool_(True),
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)


def _detect_side(path: Path) -> str:
    name = path.name.lower()
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    raise ValueError(f"无法从文件名判断左右手：{path}")


def _collect_pkls(arg_paths: list[str]) -> dict[str, Path]:
    """从命令行参数里解析出 {side: pkl_path}。"""
    found: dict[str, Path] = {}
    for raw in arg_paths:
        path = Path(raw)
        candidates: list[Path] = []
        if path.is_dir():
            candidates = sorted(
                p for p in path.rglob("*.pkl")
                if "left" in p.name.lower() or "right" in p.name.lower()
            )
        elif path.is_file() and path.suffix.lower() == ".pkl":
            candidates = [path]
        else:
            raise ValueError(f"不是有效的 pkl 文件或目录：{path}")
        for pkl in candidates:
            side = _detect_side(pkl)
            found[side] = pkl
    return found


def install(arg_paths: list[str]) -> None:
    if not arg_paths:
        raise SystemExit("--install 需要至少一个 pkl 路径或目录")

    pkls = _collect_pkls(arg_paths)
    missing = {"left", "right"} - set(pkls)
    if missing:
        raise SystemExit(f"缺少 {'/'.join(sorted(missing))} 手的 MANO pkl")

    for side in ("left", "right"):
        src = pkls[side]
        dst_pkl = MODELS_DIR / f"MANO_{side.upper()}.pkl"
        dst_npz = HAND_DIR / f"HAND_{side.upper()}_PINKY_PLUS_2MM.npz"

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_pkl)
        build_npz(src, dst_npz)
        print(f"[ok] {side}: {dst_pkl.relative_to(BUNDLE_ROOT)}")
        print(f"[ok] {side}: {dst_npz.relative_to(BUNDLE_ROOT)}")

    print("\n安装完成。现在可以运行 `python main.py` 启动。")
    status()


def status() -> None:
    pkl_left = MODELS_DIR / "MANO_LEFT.pkl"
    pkl_right = MODELS_DIR / "MANO_RIGHT.pkl"
    npz_left = HAND_DIR / "HAND_LEFT_PINKY_PLUS_2MM.npz"
    npz_right = HAND_DIR / "HAND_RIGHT_PINKY_PLUS_2MM.npz"

    items = [
        ("MANO_LEFT.pkl", pkl_left),
        ("MANO_RIGHT.pkl", pkl_right),
        ("HAND_LEFT_PINKY_PLUS_2MM.npz", npz_left),
        ("HAND_RIGHT_PINKY_PLUS_2MM.npz", npz_right),
    ]
    for label, path in items:
        mark = "已安装" if path.is_file() else "缺失"
        print(f"  {mark:<4} {label}")

    if all(path.is_file() for _, path in items):
        print("\nMANO 模型已就绪。")
    else:
        print(f"\n请先到 {MANO_DOWNLOAD_URL} 注册并下载 MANO（Models & Code，v1.2），")
        print("解压后用 `--install` 安装，或运行 `--open` 打开下载页面。")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="打开 MANO 下载页面")
    parser.add_argument("--install", nargs="+", metavar="PATH",
                        help="安装 MANO pkl（两个文件路径或一个目录）")
    args = parser.parse_args(argv)

    if args.open:
        print(f"正在打开 {MANO_DOWNLOAD_URL}")
        webbrowser.open(MANO_DOWNLOAD_URL)
        return

    if args.install:
        install(args.install)
        return

    status()


if __name__ == "__main__":
    main()
