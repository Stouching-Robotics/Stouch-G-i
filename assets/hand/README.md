# HAND+2mm 用户版模型资源

本目录随 `Stouch-G-i` 用户版一起跟踪和分发，供 21 关键点解算与 3D 显示使用。

| 文件 | 用途 |
| --- | --- |
| `HAND_{LEFT,RIGHT}_PINKY_PLUS_2MM.npz` | 左右手运行时派生网格，由 `algorithm/lite/hand.py` 加载。 |
| `models/MANO_{LEFT,RIGHT}.pkl` | 3D 显示使用的手部模型及三角面索引。 |
| [LICENSE.txt](LICENSE.txt) | 模型来源、许可及使用边界；使用前请阅读。 |

目录路径均相对于仓库根目录下的 `assets/hand/`。完整用户入口、环境和版本说明见[仓库 README](../../README.md)。

如果模型文件缺失，请恢复同一用户发行版本中的对应文件。运行时派生网格须保留 `pinky_plus_2mm` 类型和 `0.002` 米的小指延长元数据；不要混用其他版本或任意替换文件。
