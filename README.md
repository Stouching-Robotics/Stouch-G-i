# Stouch Glove 跨平台工具包（Stouch-G-i）

Stouch-G-i 是 GitHub 半开源用户发行仓库，提供 SDK／GUI 源码与受保护算法，用于设备识别、佩戴标定、21 关键点解算、压力／力值显示及已有数据回放。

SDK：`2.3.3`；GUI：`2.0.13`；文档日期：`2026-10-30`。本版为 PyArmor 9.2.8 trial 试用／测试交付。

本仓库根目录直接对应 Python SDK 根目录。所有安装和运行命令均在 `Stouch-G-i/` 内执行；桌面 EXE 由独立发行包提供。

**验证状态：**公开接口与示例已完成静态核对和模拟验证。Windows 离线解算、GUI 控件和回放检查通过；配套桌面包的 EXE 启动与合成回放检查通过。双手及左右单手实时启动已通过模拟链路检查；本版真机实时首帧尚未验收通过，动作精度与标定质量未验收。Linux 尚未测试，SDK 示例尚未连接设备验证。

## 主要功能

- USB／蓝牙 dongle 设备发现与左右手绑定。
- 单手／双手标定，单手、双手及自适应实时 3D 显示。
- 输出 `(21, 3)` 关键点（米）、`(16, 16)` 压力 ADC（计数）与拟合力值（牛顿）。
- 新佩戴标定保存补偿与对指兜底，计算后通过独立返回平展验证才可保存。
- 四指关节约束、约束 Blending 和握拳预制修正默认开启；对指修正默认关闭，Blend 初始为 `1.0`，可通过 GUI 或 SDK 调节。
- 回放已有 Parquet，渲染 MP4／PNG；通过 `sdk.open_glove()` / `sdk.open_bimanual()` 集成用户帧。

## 目录结构

```text
Stouch-G-i/
├─ README.md
├─ main.py                       # GUI 与设备管理入口
├─ replay_entry.py               # 回放辅助入口
├─ requirements.txt              # Python 3.10 依赖
├─ sdk/                          # 用户 SDK
├─ algorithm/                    # 含受保护模块的标定与解算实现
├─ common/                       # 类型、常量与资源路径
├─ gui/                          # 标定、实时显示和回放
├─ assets/                       # 模型与显示资源，含模型许可
├─ calibration/                  # canonical 双手默认文件及用户标定
├─ config/                       # 设备、标定绑定与显示配置
├─ firmware/
└─ pyarmor_runtime_000000/        # 对应平台的算法保护运行时
```

本仓库不额外嵌套 `src/`，也不包含 `release/` 或桌面 EXE。

## 环境与安装

使用 **Python 3.10 x86_64**。首次安装需要网络与 Git；`manotorch` 从固定 Git 提交安装。随包依赖包含 CPU PyTorch、NumPy、SciPy、PySide6、OpenCV、Polars 与串口库。

### Windows

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import sdk, gui; print(sdk.get_version(), gui.APP_VERSION); print(sdk.__file__)"
```

预期版本输出为 `2.3.3 2.0.13`，模块路径应位于本仓库。

如 `chumpy==0.70` 构建失败，可在已激活环境中执行后重试：

```powershell
python -m pip install numpy==1.22.4 scipy==1.11.4 six
python -m pip install --no-build-isolation --no-deps chumpy==0.70
python -m pip install -r requirements.txt
```

### Linux（安装参考，未测试）

先安装 Python 3.10、Git 和发行版需要的 Qt/XCB 依赖（例如 `libxcb-cursor0`），并配置串口权限：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import sdk, gui; print(sdk.get_version(), gui.APP_VERSION); print(sdk.__file__)"
```

保护运行时面向 Windows / Linux x86_64；随附运行时不代表 Linux 已完成验收。

## 手部模型（MANO）

关键点解算与 3D 显示使用以下随包资源：

| 文件 | 用途 |
| --- | --- |
| [HAND_LEFT_PINKY_PLUS_2MM.npz](assets/hand/HAND_LEFT_PINKY_PLUS_2MM.npz) / [HAND_RIGHT_PINKY_PLUS_2MM.npz](assets/hand/HAND_RIGHT_PINKY_PLUS_2MM.npz) | 左右手派生骨架、网格与前向运动学资源 |
| [MANO_LEFT.pkl](assets/hand/models/MANO_LEFT.pkl) / [MANO_RIGHT.pkl](assets/hand/models/MANO_RIGHT.pkl) | 手部模型资源，用于网格显示 |

请保留文件名、目录结构及 [assets/hand/LICENSE.txt](assets/hand/LICENSE.txt)。模型与派生资源须遵守随附 MANO 许可；用途范围及商业授权条件以原文为准。另见 [资源说明](assets/hand/README.md)。

## 设备绑定

插入设备后列举并登记，将占位字符串替换为设备序列号：

```powershell
python main.py devices --list
python main.py devices --set-serial left --serial "LEFT_DEVICE_SERIAL"
python main.py devices --set-serial right --serial "RIGHT_DEVICE_SERIAL"
python main.py devices --check-both
```

绑定位于 `config/glove_devices.json`；`python main.py devices --clear` 清除登记后可重新配置。`link="auto"` 同时发现 USB 与蓝牙时优先 USB。无线使用前断开手套 USB，并保持 dongle 与手套配对；蓝牙登记使用 dongle 序列号。同一串口只由一个程序占用。

## GUI 与命令行用法

| 用途 | 命令 |
| --- | --- |
| 标定选择器 → 自适应单手／双手 3D | `python main.py` |
| 标定 GUI（默认右手） | `python main.py calibration` |
| 左手标定 GUI | `python main.py calibration --side left` |
| 单手 3D 并选择标定 | `python main.py live_3d --side right --select-calibration` |
| 双手 3D 并选择标定 | `python main.py live_3d_bimanual --select-calibration` |
| 自适应 3D，复用有效标定记忆 | `python main.py live_3d_bimanual_auto` |
| 自适应 3D，强制重新选择 | `python main.py live_3d_bimanual_auto --select-calibration` |
| 回放选择器 | `python main.py replay` |
| 已有 Parquet → MP4 | `python main.py replay "recording.parquet" --out "preview.mp4"` |
| 指定帧 → PNG | `python main.py replay "recording.parquet" --frame 0` |

`recording.parquet` 为用户已有文件的占位路径。PNG 写入输入文件同目录的 `preview_frame0.png`。可调度的应用为 `devices`、`calibration`、`live_3d`、`live_3d_bimanual`、`live_3d_bimanual_auto`、`replay`。

### 标定文件如何选择

- canonical 双手默认文件为 `calibration/imu_calibration_left.json` 和 `calibration/imu_calibration_right.json`。随包文件包含佩戴校正及通过的平展验证记录，是具体佩戴的示例，应为当前手套重新标定。
- GUI 按手别筛选、按修改时间列出标定。自适应程序通过 `config/calibration_bindings.json` 记忆设备对应文件；`python main.py` 强制打开选择器。
- API 显式 `calibration` 优先；否则在 `calibration_dir`（默认本仓库 `calibration/`）下筛选当前手别的 HAND2mm JSON。存在可用 canonical 文件时，排除旧默认兼容别名，再与新增标定按修改时间取最新；时间相同时取文件名排序较大者。不递归扫描子目录。
- API 不读取 GUI 标定绑定，显式加载旧文件也不会自动改为其他配套。正式集成建议显式指定文件，避免复制／更新时间后选到另一份。

### 新标定与运行控制

按向导完成根部转动、安装平展、四次对指、握拳等步骤；计算后回到安装平展的手型与手腕朝向，静止完成独立验证，通过才可保存。返回平展只验收，不重新拟合。计算失败暂停自动采集；采集或末次验证失败可按提示静止重采，亦可取消自动采集查看原因。

旧文件保持可读，GUI 会区分佩戴校正、已有配套校正与旧标定兼容；多个配套需显式选择。旧文件读取成功不代表取得新版平展验收。

“四指关节约束”默认开启，限制食指、中指、无名指和小指的指根扭转、侧偏及关节屈伸。“对指兜底”（旧标定可能显示“对指吸附”）默认开启、Blend 100%。对新佩戴校正文件，关闭对指或将 Blend 设为 0 只关闭接触修正，佩戴补偿和末两节单轴约束继续生效。对指修正不代表完整碰撞保证或真实指尖精度验收。

## Python SDK

使用 `from sdk import open_glove, open_bimanual`。单手默认 `side="right"`、`link="auto"`、`sample_rate_hz=80`、`include_tactile=True`、`smoothing_ms=35.0`。`sample_rate_hz` 是配置默认值，不是稳定 80 Hz 保证。

用户 `HandFrame` 保持 16 个字段：

| 字段 | 含义 |
| --- | --- |
| `side`、`sequence`、`timestamp_us` | 手别、逐手帧序号、SDK 微秒时间线 |
| `joints_raw_m`、`joints_smoothed_m` | `float32 (21, 3)`，腕心关键点，米；后者进一步低通平滑 |
| `force_n` | `float32 (16, 16)` 牛顿；无拟合覆盖为 `NaN`，无扫描为 `None` |
| `tactile_adc` | `float32 (16, 16)` 原始 12-bit 计数 0–4095，无扫描为 `None` |
| `status` | 解算状态；通过 `details.get("warming_up", False)` 跳过预热占位姿态 |
| `force_fresh` | ADC／力值共享的新扫描标志；旧扫描可附在后续姿态帧中 |
| `firmware_version`、`link_kind` | 固件上报字符串（可为 `None`）与链路 |
| `device_fps`、`tactile_fps` | 姿态帧与压力扫描各自的主机到达率，Hz |
| `link_drops`、`link_errors` | 链路统计字典，按 `.get()` 读取 |
| `latency_ms` | 往返探测时延，毫秒；未知／蓝牙为 `None` |

关键点 `0` 为腕，`1–4` 为拇指，`5–8` 为食指，`9–12` 为中指，`13–16` 为无名指，`17–20` 为小指；每条指链由根到尖。两手索引顺序一致，按各自模型和标定解算。每帧以腕为平移原点，保留腕部朝向；列为 `[x, y, z]`，轴向不能直接当作屏幕坐标，接入外部引擎需核对标定与坐标变换。

`joints_raw_m` 仍包含当前解算补偿与约束，raw 只表示未作输出低通。ADC 为物理矩阵行列顺序，不因左右手重排；矩阵索引不等同于关键点索引。使用力值时处理 `None` 和 `NaN`；零值不是缺失值。滤波／积分只在新扫描更新；使用 `latest()` 或双手组合时还应逐手按序号去重。

### 单手：有限次数读取

```python
import numpy as np
from sdk import StreamTimeoutError, open_glove

with open_glove(
    "right", calibration="calibration/imu_calibration_right.json",
) as glove:
    glove.set_root_limits_enabled(True)
    glove.set_root_limits_blending_enabled(True)
    glove.set_fist_correction_enabled(True)
    glove.set_contact_controls(True, blend=0.5)
    for _ in range(100):
        try:
            frame = glove.read(timeout=0.5)
        except StreamTimeoutError:
            continue
        if frame.status.details.get("warming_up", False):
            continue
        if not np.isfinite(frame.joints_smoothed_m).all():
            continue
        print("关键点：", frame.joints_smoothed_m.shape)
        if frame.force_fresh:
            if frame.tactile_adc is not None:
                print("ADC：", frame.tactile_adc.shape)
            if frame.force_n is not None:
                values = frame.force_n[np.isfinite(frame.force_n)]
                if values.size:
                    print("有效单元平均力 (N)：", float(values.mean()))
        break
    else:
        print("本次读取未取得可用姿态帧，请检查连接和预热状态")
```

`with` 进入时启动、退出时释放。每次 `timeout` 限制等帧时间，不是连接／预热全过程总时限。`StreamTimeoutError` 可按健康状态重试；其他错误可捕获 `GloveSdkError`。`latest()` 不阻塞，首帧前可为 `None`，也可重复返回同一帧。

### 双手：逐手设置与组合读取

```python
import numpy as np
from sdk import StreamTimeoutError, open_bimanual

with open_bimanual(
    left_calibration="calibration/imu_calibration_left.json",
    right_calibration="calibration/imu_calibration_right.json",
) as gloves:
    gloves.set_root_limits_blending_enabled(True)
    gloves.set_fist_correction_enabled(True)
    gloves.left.set_contact_controls(False)
    gloves.right.set_contact_controls(True, blend=1.0)
    gloves.left.set_root_limits_enabled(True)
    gloves.right.set_root_limits_enabled(True)
    last_sequences = None
    for _ in range(100):
        try:
            frame = gloves.read(timeout=0.5)
        except StreamTimeoutError:
            continue
        hands = (frame.left, frame.right)
        sequences = tuple(hand.sequence for hand in hands)
        previous = last_sequences
        last_sequences = sequences
        if any(hand.status.details.get("warming_up", False) for hand in hands):
            continue
        if not all(np.isfinite(hand.joints_smoothed_m).all() for hand in hands):
            continue
        if previous is not None and any(
            current == old for current, old in zip(sequences, previous)
        ):
            continue
        print(frame.left.joints_smoothed_m.shape,
              frame.right.joints_smoothed_m.shape,
              "两侧帧时间差 (ms)：", frame.synchronization_error_ms)
```

一侧暂无新帧时 SDK 可能复用其最近帧；示例只消费两侧都更新的组合。各手 ADC／力值仍需分别检查序号、新鲜度及有效值。`synchronization_error_ms` 是两侧时间戳绝对差，不是硬件同步精度。时间线由主机到达时间建立起点并按设备计数增量推进，不能作精确 UTC 授时依据。

### 公开对象与设置

单手保留 `start()`、`read(timeout=None)`、`latest()`、`info()`、`health()`、`neutral_joints()`、`wait_for_firmware_version(timeout_s=5.0)`、`close()` 和上下文管理；还提供 `side`、`port`、`serial_number`、`mesh_vertices_m`、`backend` 属性。

姿态控制方法包括 `set_contact_controls(enabled: bool, blend: float = 1.0) -> None`、`set_root_limits_enabled(enabled: bool) -> None`、`set_root_limits_blending_enabled(enabled: bool) -> None` 和 `set_fist_correction_enabled(enabled: bool) -> None`。Blend 须为 `[0, 1]` 内有限数，否则抛出 `ValueError`。默认四指约束、约束 Blending 和握拳预制修正开启；对指关闭，Blend 初始为 `1.0`。

约束 Blending 沿本次标定的对指动作线性减弱四指约束，关闭后恢复固定约束。握拳预制修正在有效的本次佩戴握拳采样附近修正四指，拇指保持原解算活动；缺少有效握拳采样的旧标定不启用该修正。设置作用于后续解算，已排队帧可能仍反映之前设置。

双手保留 `start()`、`read(timeout=None)`、`health()`、`close()`、上下文管理及 `left` / `right` 单手对象；也可调用双手对象的同名设置方法统一设置两侧。SDK 版本查询为 `sdk.get_version()`，GUI 为 `gui.APP_VERSION`，固件用 `info().firmware_version` 读取并通过 `display_firmware_version()` 显示、`firmware_real_version()` 比较；未知固件为 `None`，不能用 SDK 版本代填。

## 保护与验证状态

部分算法采用 PyArmor 代码混淆与运行时保护，须保留 `pyarmor_runtime_000000/`。本版为 **PyArmor 9.2.8 trial 试用／测试版本**，已通过上述 Windows 离线检查。保护不等于模型、配置及全部资源加密，模型许可见随包 LICENSE。

公开用户接口提供关键点、ADC、力值和诊断，不提供原始 IMU 流、逐传感器磁力计状态、管理员拟合参数读写或 SDK 原始录制方法。回放入口使用用户已有会话。

## 常见问题

- **依赖或保护运行时加载失败**：核对 Python 3.10 x86_64、依赖、运行时目录及 `sdk.__file__`。
- **找不到设备或双手无法启动**：运行 `python main.py devices --list`，检查左右手登记、链路、标定手别及串口占用。
- **自动标定选择变化**：按修改时间核对候选，集成时显式传入 canonical 或自采文件；API 不读取 GUI 标定绑定。
- **无法保存新标定**：完成计算和最后返回平展验证，按界面提示重采。
- **预热后姿态仍异常**：联合检查 `warmup_timed_out` 和 `health()`，确认佩戴与所用标定匹配。
- **出现 `[WARNING][MAG]`**：新上报通道的磁力计校准从等级 3 下降，检查磁环境并缓慢画圆重新校准。没有警告不代表全部传感器都已合格。
- **压力或力值重复／缺失**：逐手去重并检查 `force_fresh`、`None`、`np.isfinite()` 和 `tactile_fps`。
- **Linux 串口／Qt 问题**：配置串口访问权限及 Qt/XCB 依赖；本轮未完成 Linux 验收。

反馈时提供版本、链路与健康信息，发送日志前隐去设备序列号和个人路径。
