# Stouch Glove 跨平台工具包（Stouch-G-i）

Stouch-G-i 是 GitHub 半开源用户发行仓库，提供 SDK/GUI 源码与受保护的算法模块，用于数据手套设备识别、姿态标定、21 关键点求解、触觉显示及已有数据回放。

当前 SDK 版本：`2.3.1`；GUI 版本：`2.0.11`。

本仓库提供可直接运行的用户版源码，桌面 EXE 使用独立发行包。

## 主要功能

- 设备发现与左右手绑定，支持 USB / 蓝牙链路自动选择。
- GUI 单手/双手 IMU 标定，单手、双手及自适应双手 3D 实时显示。
- 输出 21×3 手部关键点（米）、16×16 触觉 ADC 和拟合力值（牛顿）。
- 回放已有 Parquet 数据，并渲染 MP4 / PNG。
- 通过 `sdk.open_glove` / `sdk.open_bimanual` 集成用户帧。

## 目录结构

所有安装和运行命令均在 `Stouch-G-i` 仓库根目录执行：

```text
Stouch-G-i/
├── README.md
├── .gitignore
├── main.py                       # GUI 与设备管理统一入口
├── replay_entry.py               # 回放辅助入口
├── requirements.txt              # Python 3.10 依赖
├── sdk/                          # 用户 SDK
│   └── io/                       # 内部通信、协议与数据文件处理
├── algorithm/                    # 含受保护模块的标定与求解算法
├── common/                       # 公共类型、常量和路径工具
├── gui/                          # 标定、实时显示和回放
├── assets/                       # 手部模型、几何与显示资源
├── calibration/                  # 随包标定及用户标定结果
├── config/                       # 设备绑定、标定绑定和显示配置
├── firmware/                     # 固件文件
└── pyarmor_runtime_000000/        # Windows / Linux x86_64 保护运行时
```

## 环境与安装

使用 **Python 3.10 x86_64**。首次安装需要网络和 Git；`manotorch` 从固定 GitHub commit 安装。依赖包括 CPU 版 PyTorch、NumPy、SciPy、PySide6、OpenCV、Polars 和串口库。

### Windows

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sdk, gui; print(sdk.get_version(), gui.APP_VERSION)"
```

如 `chumpy==0.70` 构建失败，在已激活的环境中执行后重试：

```powershell
python -m pip install numpy==1.22.4 scipy==1.11.4 six
python -m pip install --no-build-isolation --no-deps chumpy==0.70
python -m pip install -r requirements.txt
```

### Linux

先安装 Python 3.10、Git 和对应发行版的 Qt/XCB 运行依赖（如 `libxcb-cursor0`）：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sdk, gui; print(sdk.get_version(), gui.APP_VERSION)"
```

请按部署环境配置串口设备权限。包内包含 Linux x86_64 保护运行时；Linux 尚未实机验收。Windows 与 Linux 使用相同的 `python main.py ...` 命令。

## 手部模型（MANO）

21 关键点求解与 3D 显示使用随包分发的 MANO 模型及派生资源：

| 文件 | 用途 |
|---|---|
| [MANO_LEFT.pkl](assets/hand/models/MANO_LEFT.pkl) / [MANO_RIGHT.pkl](assets/hand/models/MANO_RIGHT.pkl) | 官方模型资源 |
| [HAND_LEFT_PINKY_PLUS_2MM.npz](assets/hand/HAND_LEFT_PINKY_PLUS_2MM.npz) / [HAND_RIGHT_PINKY_PLUS_2MM.npz](assets/hand/HAND_RIGHT_PINKY_PLUS_2MM.npz) | 求解与显示使用的派生网格 |

MANO 由 Max-Planck 提供，许可限非商业科研/教育等用途；完整条款见 [assets/hand/LICENSE.txt](assets/hand/LICENSE.txt)。商业用途请联系 Max-Planck（ps-license@tue.mpg.de）。

## 设备绑定

插入手套后查看设备，将下面的占位字符串替换为对应设备的 USB 序列号：

```powershell
python main.py devices --list
python main.py devices --set-serial left --serial "LEFT_USB_SERIAL"
python main.py devices --set-serial right --serial "RIGHT_USB_SERIAL"
python main.py devices --check-both
```

绑定保存在 `config/glove_devices.json`。需要重新绑定时，可用 `python main.py devices --clear` 清空记忆，再按上述步骤设置。

## GUI 与命令行用法

`main.py` 的应用白名单为 `devices`、`calibration`、`live_3d`、`live_3d_bimanual`、`live_3d_bimanual_auto`、`replay`。

| 功能 | 命令 |
|---|---|
| 标定选择器 → 自适应单手/双手 3D | `python main.py` |
| 标定 GUI（默认右手） | `python main.py calibration` |
| 左手标定 GUI | `python main.py calibration --side left` |
| 单手 3D，并选择标定 | `python main.py live_3d --side right --select-calibration` |
| 双手 3D，并选择标定 | `python main.py live_3d_bimanual --select-calibration` |
| 自适应实时 3D，复用有效标定记忆 | `python main.py live_3d_bimanual_auto` |
| 自适应实时 3D，强制重选标定 | `python main.py live_3d_bimanual_auto --select-calibration` |
| 回放文件选择器 | `python main.py replay` |
| 将已有 Parquet 渲染为 MP4 | `python main.py replay "recording.parquet" --out "preview.mp4"` |
| 将指定帧渲染为 PNG | `python main.py replay "recording.parquet" --frame 0` |

回放示例中的 `recording.parquet` 是用户提供的已有文件；请替换为实际路径。PNG 输出固定为输入文件同目录的 `preview_frame0.png`。

### 标定文件如何选择

- GUI 选择器按左右手筛选兼容文件，并按修改时间从新到旧列出。自适应实时程序通过 `config/calibration_bindings.json` 记忆设备序列号对应的标定；`python main.py` 会强制进入选择器。
- GUI 未指定文件/标定目录、未经过选择且无有效标定绑定时，文件回退值为左手 `calibration/imu_2d_calibration.json`、右手 `calibration/imu_calibration_right_default.json`。
- API 省略标定参数时，在仓库 `calibration/`（或 `calibration_dir`）内按左右手和 HAND2mm 兼容性筛选，取修改时间最新的文件；时间相同再按文件名排序取最大值。API 不使用 GUI 的标定绑定。
- 随包提供多份标定。建议使用 GUI 生成适配当前手套的文件，并在集成时显式指定，避免文件修改时间变化影响选择。以下示例显式选用随包文件，实际使用请换成对应手套的标定。

## Python SDK

推荐从 `sdk` 导入 `open_glove` / `open_bimanual`。单手默认 `side="right"`、`link="auto"`、`sample_rate_hz=80`、`include_tactile=True`、`smoothing_ms=35.0`。可用 `calibration` 显式指定文件，或用 `calibration_dir` 指定扫描目录。

用户版 `HandFrame` 字段如下：

| 字段 | 含义 |
|---|---|
| `side`、`sequence`、`timestamp_us` | 左右手、帧序号与微秒时间戳 |
| `joints_raw_m`、`joints_smoothed_m` | 未平滑/已平滑的关键点，均为 `(21, 3)`，单位米 |
| `tactile_adc` | `(16, 16)` 原始 12-bit 触觉 ADC，计数范围 0–4095；无扫描时为 `None` |
| `force_n` | `(16, 16)` 拟合力值，单位牛顿；未拟合区域为 `NaN`，无扫描时为 `None` |
| `force_fresh` | 上述 ADC 与力值共用的新鲜度标志，`False` 时可能沿用之前的扫描 |
| `status` | 求解状态；`details["warming_up"]` 为真时，关键点为预热占位姿态 |
| `firmware_version`、`link_kind` | 固件版本（可能为 `None`）与当前链路类型 |
| `device_fps`、`tactile_fps` | 姿态帧与触觉扫描各自的到达帧率 |
| `link_drops`、`link_errors` | 链路丢帧与错误统计字典 |
| `latency_ms` | 延迟估计，可能为 `None` |

`force_fresh` 同时适用于 `tactile_adc` 与 `force_n`；累计或滤波时只处理新扫描，并跳过非有限力值。SDK 版本与固件版本独立，SDK 版本可用 `sdk.get_version()` 查询。

### 单手：在有限时间内读取一帧

```python
from time import monotonic
import numpy as np
from sdk import StreamTimeoutError, open_glove

with open_glove("right", calibration="calibration/imu_default_right.json") as glove:
    deadline = monotonic() + 10.0
    while monotonic() < deadline:
        try:
            frame = glove.read(timeout=0.5)
        except StreamTimeoutError:
            continue
        if frame.status.details.get("warming_up", False):
            continue
        if not np.isfinite(frame.joints_smoothed_m).all():
            continue
        print("关键点形状：", frame.joints_smoothed_m.shape)
        if frame.force_fresh:
            if frame.tactile_adc is not None:
                print("触觉 ADC 形状：", frame.tactile_adc.shape)
            if frame.force_n is not None:
                values = frame.force_n[np.isfinite(frame.force_n)]
                if values.size:
                    print("有效力值均值（N）：", float(values.mean()))
        break
    else:
        print("读取阶段 10 秒内未取得可用姿态帧")
```

`read(timeout=...)` 超时会抛出 `StreamTimeoutError`；单手 `latest()` 不阻塞，首帧前可能返回 `None`。`health()` 可查看连接状态，`info()` 可查看版本与链路信息。`with` 退出时释放串口与采集资源。

### 双手：读取同步关键点

```python
from time import monotonic
import numpy as np
from sdk import StreamTimeoutError, open_bimanual

with open_bimanual(
    left_calibration="calibration/imu_default_left.json",
    right_calibration="calibration/imu_default_right.json",
) as gloves:
    deadline = monotonic() + 10.0
    while monotonic() < deadline:
        try:
            frame = gloves.read(timeout=0.5)
        except StreamTimeoutError:
            continue
        hands = (frame.left, frame.right)
        if any(h.status.details.get("warming_up", False) for h in hands):
            continue
        if not all(np.isfinite(h.joints_smoothed_m).all() for h in hands):
            continue
        print(frame.left.joints_smoothed_m.shape, frame.right.joints_smoothed_m.shape)
        print("同步误差（ms）：", frame.synchronization_error_ms)
        break
    else:
        print("读取阶段 10 秒内未取得可用双手姿态帧")
```

双手帧的 `left` / `right` 均为上述用户版 `HandFrame`；读取两侧触觉或力值时，分别检查各自的 `force_fresh`、`None` 与非有限值。

## 保护与验证状态

部分算法模块使用 **PyArmor 9.2.8 trial** 保护；运行所需的 Windows / Linux x86_64 运行时已包含在仓库。模型使用边界见随附许可证。

Windows 离线验证已通过；Linux 仅随包提供运行时，尚未真机验收，手套硬件未测。

## 常见问题

- **依赖无法加载**：确认使用 Python 3.10 x86_64，且在当前虚拟环境完整安装 `requirements.txt`。
- **找不到手套/双手无法启动**：运行 `python main.py devices --list` 核对设备，再检查左右手绑定及所选标定文件。
- **Linux 串口权限不足**：按部署环境配置串口访问权限，并重新连接设备。
- **Qt 缺少 `libxcb-cursor.so.0`**：安装对应发行版的 Qt/XCB 依赖（如 `libxcb-cursor0`）。
