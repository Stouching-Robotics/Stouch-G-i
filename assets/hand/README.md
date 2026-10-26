# HAND+2mm 运行资源

本用户包随附以下运行资源，使用前请阅读并遵守 [LICENSE.txt](LICENSE.txt)。

| 文件 | 用途 |
| --- | --- |
| `HAND_LEFT_PINKY_PLUS_2MM.npz` / `HAND_RIGHT_PINKY_PLUS_2MM.npz` | 左右手 +2 mm 派生骨架、网格与前向运动学 |
| `models/MANO_LEFT.pkl` / `models/MANO_RIGHT.pkl` | 三维显示所需三角面索引 |

请保留文件名和目录结构。缺少 NPZ 会阻止关键点解算；缺少 PKL 会使三维手部网格显示降级。
算法会核验 NPZ 中的派生模型标识及 0.002 m 小指加长元数据。
资源随包附带不改变原模型及派生资源的许可条件。
