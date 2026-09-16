# Stouch Glove 跨平台工具包

本工具包用于 STM32 数据手套的设备识别、IMU/触觉数据采集、姿态标定、21 关键点求解、3D 实时显示、录制与回放。

当前 SDK 版本：`2.0.3`

> 支持平台：**Windows x86_64 / Linux x86_64 + Python 3.10**

## 主要功能

- 自动识别左右手套的 STM32 USB CDC 串口。
- 读取 16 路 IMU 四元数及 16×16 触觉压力矩阵。
- 支持单手/双手 IMU 标定。
- 将 IMU 数据解算为 21×3 手部关键点（单位：米）。
- 提供单手、双手及自适应双手 3D 显示。
- 支持 Parquet 录制与 MP4/PNG 回放渲染。
- 提供 Python SDK 接口，便于集成到其他项目。

## 目录结构

```text
SDK/
├── main.py                       # Windows/Linux 统一入口
├── requirements.txt             # Python 3.10 依赖
├── download_mano.py             # 下载并安装 MANO 手部模型
├── 99-stm32-glove.rules         # Linux udev 串口权限规则
├── runtime/                     # 对外 Python SDK 接口
├── glove_io/                    # 串口、协议、录制与回放
├── algorithm/                   # 标定与求解核心
├── gui/                         # 标定、实时显示和回放程序
├── common/                      # 公共数据类型、常量和路径工具
├── config/                      # 设备、显示和运行配置
├── calibration/                 # 默认标定文件及用户标定结果
├── assets/                      # 手部模型、几何与显示资源
└── firmware/                    # USB/蓝牙固件
```

`data/` 不是预置目录；录制数据后会自动创建。

## 环境要求

- 必须使用 **Python 3.10 x86_64**。
- 首次安装需要网络和 Git；`manotorch` 会从指定的 GitHub commit 安装。
- 安装内容包含 CPU 版 PyTorch、NumPy、SciPy、PySide6、OpenCV、Polars 和串口库。
- 必须在工具包的 `SDK` 目录中运行下面的命令。

### Windows

建议将虚拟环境建在工具包之外：

```powershell
py -3.10 -m venv C:\venvs\stouch_glove
C:\venvs\stouch_glove\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import runtime; print(runtime.get_version())"
```

如 `chumpy==0.70` 构建失败：

```powershell
pip install numpy==1.22.4 scipy==1.11.4 six
pip install --no-build-isolation --no-deps chumpy==0.70
pip install -r requirements.txt
```

Windows 下使用 `python main.py ...` 启动各项功能。

### Linux

先确保系统已安装 Python 3.10、Git 和 Qt 所需的 `libxcb-cursor0`，然后执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import runtime; print(runtime.get_version())"

sudo cp 99-stm32-glove.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

安装 udev 规则后请重新插拔手套。Linux 与 Windows 使用相同的 `python main.py ...` 命令。

## 手部模型（MANO）

21 关键点求解与 3D 显示依赖 MANO 手部模型。MANO 由 Max-Planck 提供，仅授权**非商业**科研/教育用途，且**禁止再分发**，因此本仓库不包含模型文件，需每位使用者自行注册下载。

1. 访问 <https://mano.is.tue.mpg.de/>，注册账号并接受许可协议。
2. 下载 **Models & Code（MANO v1.2）**，解压后得到 `MANO_LEFT.pkl` 与 `MANO_RIGHT.pkl`。
3. 在本（`SDK`）目录运行安装脚本：

   ```powershell
   python download_mano.py --install MANO_LEFT.pkl MANO_RIGHT.pkl
   ```

   也可以把解压后的整个目录直接传给 `--install`，脚本会自动识别左右手文件。

脚本会把两个 PKL 复制到 `assets/hand/models/`，并据此重建 `assets/hand/HAND_*_PINKY_PLUS_2MM.npz`（求解与 3D 显示实际加载的是后者）。运行 `python download_mano.py` 可随时查看安装状态，`--open` 可直接打开下载页面。许可全文见 `assets/hand/LICENSE.txt`；商业用途请单独联系 Max-Planck（ps-license@tue.mpg.de）。

## 设备绑定

先插入手套并查看系统识别到的设备：

```powershell
python main.py devices --list
```

将输出的 USB 序列号绑定为左手或右手：

```powershell
python main.py devices --set-serial left  --serial <LEFT_SERIAL>
python main.py devices --set-serial right --serial <RIGHT_SERIAL>
python main.py devices --check-both
```

清空所有已记住序列号：

```powershell
python main.py devices --clear
```

设备信息保存在 `config/glove_devices.json`。

## 命令行用法

Windows 和 Linux 的命令相同：

| 功能 | 命令 |
|---|---|
| 自适应单手/双手 3D | `python main.py` |
| 强制重新选择标定 | `python main.py --select-calibration` |
| 标定 GUI（默认右手） | `python main.py calibration` |
| 左手标定 GUI | `python main.py calibration --side left` |
| 单手 3D | `python main.py live_3d --side right` |
| 双手 3D | `python main.py live_3d_bimanual --select-calibration` |
| 自适应双手 3D | `python main.py live_3d_bimanual_auto --select-calibration` |
| 单手 IMU 终端 | `python main.py imu_live` |
| 双手 IMU 终端 | `python main.py imu_live_bimanual` |
| 触觉矩阵 | `python main.py tactile --side right` |
| 回放选择器 | `python main.py replay` |
| 指定 Parquet 回放 | `python main.py replay <PARQUET>` |

默认标定文件：

- 右手：`calibration/imu_calibration.json`
- 左手：`calibration/imu_2d_calibration.json`

实时窗口内的录制会写入 `data/keypoints_21*/<SESSION>/`。

## Python SDK

所有公开接口都可以直接从 `runtime` 导入：

```python
from runtime import (
    DeviceManager,
    RawImuStream,
    TactileStream,
    HandSolver,
    Glove,
    GloveConfig,
    BimanualGlove,
    BimanualConfig,
    ImuCalibrator,
    RecordingReplay,
)
```

### 接口总览

| 接口 | 用途 | 主要返回值 |
|---|---|---|
| `DeviceManager` | 查找手套、绑定左右手、解析串口 | `DeviceInfo`、`DeviceBindings` |
| `RawImuStream` | 读取 STM32 原始 16 路 IMU 数据 | `RawImuFrame` |
| `TactileStream` | 读取 16×16 触觉压力矩阵 | `TactileFrame` |
| `HandSolver` | 将一帧原始 IMU 数据解算为 21 关键点 | `KeypointFrame` |
| `Glove` | 单手采集、解算、触觉和录制的一体化接口 | `HandFrame` |
| `BimanualGlove` | 同步读取左右手并支持双手录制 | `BimanualFrame` |
| `ImuCalibrator` | 按步骤完成单只手套的 IMU 标定 | `CalibrationProgress` |
| `RecordingReplay` | 检查 Parquet 录制文件并渲染 MP4 | `RecordingInfo`、`ReplayResult` |

### 主要数据结构

| 数据结构 | 重要字段 | 数据形状或含义 |
|---|---|---|
| `RawImuFrame` | `quaternions_xyzw` | `(16, 4)`，原始四元数顺序为 XYZW |
| `RawImuFrame` | `present_mask`、`valid_mask` | `(16,)`，IMU 是否存在及数据是否有效 |
| `TactileFrame` | `samples` | `(16, 16)`，触觉压力矩阵 |
| `KeypointFrame` | `joints_m` | `(21, 3)`，21 个关键点，单位为米 |
| `HandFrame` | `joints_raw_m`、`joints_smoothed_m` | `(21, 3)`，原始及平滑后的关键点 |
| `HandFrame` | `imu_xyzw`、`imu_valid`、`tactile` | IMU 姿态、有效标记及可选触觉矩阵 |
| `BimanualFrame` | `left`、`right` | 左右手各一个 `HandFrame` |
| `BimanualFrame` | `synchronization_error_ms` | 左右手帧的时间同步误差，单位为毫秒 |

### 1. DeviceManager：设备发现与绑定

默认使用 `config/glove_devices.json` 保存绑定关系。`resolve_port()` 返回 `(串口名称, USB序列号)`。

```python
from runtime import DeviceManager

devices = DeviceManager()

# 查看当前连接的手套
for item in devices.list_devices():
    print(item.device, item.serial_number, item.bound_side)

# 首次使用时，根据实际序列号完成绑定
devices.bind("right", "YOUR_RIGHT_USB_SERIAL")

# 后续无需依赖固定的 COM 口或 /dev/tty* 名称
port, serial_number = devices.resolve_port("right")
print(port, serial_number)

# 双手程序启动前可检查左右手是否都已连接
bindings = devices.validate_bimanual()
print(bindings.left_port, bindings.right_port)
```

常用方法：

- `list_devices()`：列出当前连接的 STM32 手套。
- `bind(side, serial_number)`：将 USB 序列号绑定到 `left` 或 `right`。
- `resolve_port(side)`：根据绑定关系找到当前串口。
- `get_bindings(resolve_ports=True)`：读取绑定关系，并可同时解析串口。
- `validate_bimanual()`：确认左右手绑定和连接状态可用于双手模式。
- `swap_bindings()`：交换左右手绑定。

### 2. RawImuStream 与 TactileStream：原始传感器数据

下面的示例让 IMU 和触觉共用同一个串口连接：

```python
from runtime import DeviceManager, RawImuStream

port, _ = DeviceManager().resolve_port("right")

with RawImuStream(serial_port=port) as imu_stream:
    tactile_stream = imu_stream.tactile_stream()

    imu_frame = imu_stream.read(timeout=3.0)
    tactile_frame = tactile_stream.read(timeout=3.0)

    print(imu_frame.quaternions_xyzw.shape)  # (16, 4)
    print(imu_frame.valid_mask.shape)        # (16,)
    print(tactile_frame.samples.shape)       # (16, 16)
```

两种数据来自同一个 USB CDC 串口。需要同时读取时，应使用 `imu_stream.tactile_stream()` 共享连接，不要针对同一只手套分别创建两个独立串口对象。

数据读取方式：

- `read(timeout=...)`：阻塞等待一帧，超时会抛出 `StreamTimeoutError`。
- `poll(maximum=...)`：立即返回缓冲区中已有的帧，不阻塞。
- `frames(timeout=...)`：持续迭代数据帧，适合采集循环。
- `status`、`connected`、`last_error`：查看 IMU 数据流的连接状态。

### 3. HandSolver：离线解算 21 关键点

`HandSolver` 不负责打开串口，它只负责把 `RawImuFrame` 或 NumPy 四元数数组转换为手部关键点，因此也可以处理已保存的 IMU 数据。

```python
from runtime import HandSolver

solver = HandSolver(
    side="right",
    calibration="calibration/imu_calibration.json",
)

# imu_frame 是 RawImuStream.read() 返回的 RawImuFrame
keypoints = solver.process(imu_frame)
print(keypoints.joints_m.shape)  # (21, 3)，单位：米
print(keypoints.valid_mask)      # 16 路 IMU 有效状态
```

也可以使用 `solve(imu_xyzw, valid_mask)` 直接传入形状为 `(16, 4)` 的 XYZW 四元数数组。切换到一段新的、不连续的数据流时，可调用 `reset_stream_state()` 清除内部连续帧状态。

### 4. Glove：推荐的单手高级接口

一般的单手应用建议直接使用 `Glove`。它会根据设备绑定打开对应手套，并把 IMU、21 关键点、触觉和状态放在同一个 `HandFrame` 中。

```python
from runtime import Glove, GloveConfig

config = GloveConfig(
    side="right",
    calibration="calibration/imu_calibration.json",
    include_tactile=True,
)

with Glove(config) as glove:
    frame = glove.read(timeout=3.0)
    print(frame.joints_smoothed_m.shape)  # (21, 3)
    if frame.tactile is not None:
        print(frame.tactile.shape)        # (16, 16)
```

`GloveConfig` 的常用参数：

- `side`：`"left"` 或 `"right"`。
- `calibration`：该手套对应的标定 JSON。
- `serial_number`：可选；指定 USB 序列号。省略时使用设备绑定。
- `sample_rate_hz`：目标采样率，默认 `80`。
- `include_tactile`：是否在输出帧中包含触觉数据，默认开启。
- `smoothing_ms`：关键点平滑窗口，默认 `35.0` 毫秒。
- `recording_root`：可选的录制输出目录。

`Glove` 还提供 `latest()`、`health()`、`start_recording()`、`stop_recording()` 和 `recording_status`。推荐使用 `with`，确保退出时关闭串口和采集线程。

### 5. BimanualGlove：双手同步接口

```python
from runtime import BimanualConfig, BimanualGlove, GloveConfig

left = GloveConfig(
    side="left",
    calibration="calibration/imu_2d_calibration.json",
)
right = GloveConfig(
    side="right",
    calibration="calibration/imu_calibration.json",
)

with BimanualGlove(BimanualConfig(left=left, right=right)) as gloves:
    frame = gloves.read(timeout=3.0)
    print(frame.left.joints_smoothed_m.shape)   # (21, 3)
    print(frame.right.joints_smoothed_m.shape)  # (21, 3)
    print(frame.synchronization_error_ms)
```

`health()` 返回左右手各自的连接状态和同步误差。双手录制可使用 `start_recording(name)`、`stop_recording()` 和 `recording_status`。

### 6. ImuCalibrator：程序化标定

`ImuCalibrator` 将标定过程拆分为多个步骤。应用程序应展示每一步的 `title`，提示用户完成对应动作后再调用 `run_step()`。

```python
from runtime import ImuCalibrator

with ImuCalibrator(side="right") as calibrator:
    health = calibrator.connect()
    if not health.connected:
        raise RuntimeError(health.last_error or health.message)

    calibrator.start_new()
    for step in calibrator.steps:
        input(f"请完成动作：{step.title}，然后按回车继续...")
        result = calibrator.run_step(step.id)
        if not result.ok:
            raise RuntimeError(result.message)

    output = calibrator.save(
        "right_calibration.json",
        directory="calibration",
        overwrite=False,
    )
    print(output)
```

如果某一步失败，可调整手势后调用 `retry_step(step.id)`。`progress()` 返回总步骤数、已完成步骤和是否全部完成。

### 7. RecordingReplay：录制文件检查与回放

```python
from runtime import RecordingReplay

replay = RecordingReplay()
info = replay.inspect("data/example/chunk-000.parquet")
print(info.mode, info.sides, info.frame_count, info.sample_fps)
print(info.has_imu, info.has_tactile)

result = replay.render_mp4(
    "data/example/chunk-000.parquet",
    "data/example/preview.mp4",
    view="static",          # 也可以使用 "rotating"
    smooth=True,
    include_tactile=True,
    overwrite=True,
)
print(result.output_path, result.frame_count)
```

### 异常处理与资源释放

常用异常也从 `runtime` 导出，例如：

```python
from runtime import DeviceNotFoundError, StreamTimeoutError

try:
    frame = glove.read(timeout=1.0)
except StreamTimeoutError:
    print("等待数据超时")
except DeviceNotFoundError as exc:
    print(f"未找到手套：{exc}")
```

`RawImuStream`、`TactileStream`、`Glove`、`BimanualGlove` 和 `ImuCalibrator` 都支持 `with` 上下文管理。长期运行的程序也可以显式调用对应的 `close()` 或 `stop()` 方法。

## 发布注意事项

- 正式打包前请删除所有 `__pycache__/`、`*.pyc`、录制数据和临时标定文件。
- 请检查 `config/glove_devices.json` 与 `config/calibration_bindings.json`，避免对外发布无关设备序列号或失效路径。
- `assets/hand/` 中的 MANO 模型及派生资源受单独许可证约束，对外发布前请阅读 `assets/hand/LICENSE.txt` 并确认再分发授权。

## 常见问题

- **Python 依赖无法加载**  
  确认当前是 Python 3.10 x86_64，并已在当前虚拟环境中完整安装 `requirements.txt`。

- **Windows 找不到手套**  
  运行 `python main.py devices --list`，确认设备出现为 `COMx`，然后绑定 USB 序列号。

- **Linux 串口权限不足**  
  安装 `99-stm32-glove.rules` 并重新插拔手套。

- **Qt 报错缺少 `libxcb-cursor.so.0`**  
  安装系统包 `libxcb-cursor0`。

- **双手模式无法启动**  
  先运行 `devices --check-both`，然后确认左右手都有对应的标定 JSON。
