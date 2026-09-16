# STM32 手套固件 — 蓝牙版

本 SDK 对接的硬件：STM32 灵巧手，16×BNO055 IMU + 16×16 触觉压力矩阵 → CH9438 → USB CDC / 蓝牙 SPP。

当前固件版本：**v2.0.2**（蓝牙版）

## 变更说明

- 帧结构瘦身：帧头 18→14 字节，删 flags 字段，valid_mask 移入 IMU payload（130B）。
- 矩阵帧 12-bit 打包（524→384B），蓝牙空中带宽约 41.5→32.8 KB/s，传输更稳。
- 四元数 int16 Q14 直通，去掉 float 往返。
- seq 最高位 = 磁力计就绪标志；无效帧补包（连续掉帧 ≤5 填上一帧四元数）。
- 上位机（imu_monitor / matrix_gui）同步适配 14 字节帧头。

## 无线架构

```
手套: STM32 ──UART──▶ FSC-BT9104 (从机, 广播名 "traphand")
                        │  SPP 空中
                        ▼
PC: BP101Y dongle (主机) ──USB CDC──▶ COM 口 (VID 0483 PID 2013)
```

## 文件

> 本目录只分发原始 `.bin` 镜像，不分发 Intel HEX。用 STM32CubeProgrammer 等工具烧录 `.bin` 时，
> 手动填写下表中的起始地址即可，与 `.hex` 烧录等效。

| 文件 | 烧录地址 | 说明 |
|---|---|---|
| `bootloader.bin` | `0x08000000` (32KB) | Bootloader，左右手共用，出厂烧一次 |
| `left/stm32_imu_usb.bin` | `0x08008000` (96KB) | APP 固件（左手序） |
| `right/stm32_imu_usb.bin` | `0x08008000` (96KB) | APP 固件（右手序） |

## 烧录顺序

1. 先烧 `bootloader.bin`（地址 `0x08000000`）
2. 再烧对应手的 APP：`left/` 或 `right/` 下的 `stm32_imu_usb.bin`（地址 `0x08008000`）

> 若设备出厂已烧好 bootloader，只需烧 APP 即可。

## 蓝牙连接（换机）

绑定完成后，只需**拔插 dongle** 即可实现换机连接（手套重启自动连 dongle，无需重新配对）。

## 待测试 / 可优化

- **未测试内容**：蓝牙最佳使用距离。
- **可优化方向**：根据左右手压力触点不同，降低数据量。
