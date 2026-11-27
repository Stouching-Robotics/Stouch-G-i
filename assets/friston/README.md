# Friston 左右手骨架和蒙皮

每侧提供一个 NPZ 和一个 USDA，各有 16 个关节、326 个顶点和 641 个三角面。模型来自 G-i Toolkit 的 `houdini-static-v2`，由 Sintaka 使用 Tripo3d 生成后在 Houdini 中雕刻、调整骨架和权重。源文件 `apps/assets/hands/left_hand.npz` 的 SHA-256 为 `b53cafe4d52c6218abc2c7268ddb77e2ecc9d64151bc0303ff00f7173c7e8d19`。

## 姿态和坐标

- 关节顺序为腕、拇指、食指、中指、无名指、小指。每指由近端到远端各三个关节。单位为米，四元数顺序为 XYZW。第 0 项是腕根的绝对朝向，其余各项是相对于父关节的完整朝向，可直接和静息局部平移组成 FK 变换。
- 腕根静息朝向为单位旋转，原点位于腕。静态公共坐标系的 +Z 指向指端，+Y 指向手背，+X = +Y × +Z。手指各自的局部 +Z 指向下一关节或指尖，+Y 取参考骨骼的局部 +Y 在垂直于 +Z 平面上的归一化投影，+X = +Y × +Z。
- 21 个关键点沿同一顺序排列，每指第四点是指尖。指尖由远节关节和五个固定的局部偏移生成。右手沿公共 X 轴反射位置，旋转用反射共轭，三角面反转绕序。
- 源资产已将雕刻网格重绑到 `static-v2` 骨架。导出使用 `source_rest` 计算静息变换，并使用 `source_bind` 计算蒙皮绑定，两者采用相同的关节换基。原捕获表保留较早的绑定轴和位置，其中拇指根比当前绑定相差约 9.14 mm。前臂权重并入腕根，虚拟末端随远节运动。

## Python

```python
from sdk.friston import load_skeleton

hand = load_skeleton("left")
pose = hand.rest_local_xyzw  # 实时数据使用相应手的 frame.joint_xyzw
points = hand.keypoints_m(pose)
vertices = hand.mesh_vertices_m(pose)
triangles = hand.faces
```

NPZ 可用 `numpy.load(path, allow_pickle=False)` 读取。主要字段如下。

| 字段 | 内容 |
| --- | --- |
| `joint_names`、`joint_paths`、`parents` | 同一顺序的关节名称、路径和父节点索引，腕的父索引为 -1 |
| `rest_local_offsets_m`、`rest_local_xyzw` | `(16, 3)` 静息局部平移与 `(16, 4)` 静息局部旋转 |
| `bind_world_m` | `(16, 4, 4)` 骨架空间绑定矩阵，作用于列向量 |
| `tip_offsets_m` | `(5, 3)` 远节局部指尖偏移，顺序为拇、食、中、环、小 |
| `vertices_m`、`normals`、`faces` | 绑定网格顶点、顶点法线和三角索引 |
| `skin_joint_indices`、`skin_weights` | 每顶点的关节索引和线性混合权重，空槽权重为零 |

`hand.source_world_to_local_xyzw(world)` 将参考 Friston 的 `(16, 3, 3)` 绝对关节旋转转换为公共姿态。输入是与手侧匹配的参考显示世界（+Y 指端、+Z 手背，右手已经做过 X 反射共轭），骨骼重排为腕、拇、食、中、环、小。参考和公共世界之间的矩阵为 `[[-1, 0, 0], [0, 0, 1], [0, 1, 0]]`。该函数再用每个关节固定的静息换基转换局部轴。

## OpenUSD

在 `/Hand/Skeleton` 和 `/Hand/Mesh` 读取标准 UsdSkel 骨架及线性混合蒙皮。场景单位为米，Y 向上。UsdSkel 使用行向量矩阵，`restTransforms` 是局部变换，`bindTransforms` 是骨架空间绑定变换。五个指尖的父路径和局部偏移记录在骨架的 `friston:tipJoints` 与 `friston:tipOffsets` 属性中。参见 [OpenUSD UsdSkel 骨架规范](https://openusd.org/dev/api/_usd_skel__schema_overview.html)。

应用可绑定 `UsdSkelAnimation`，按 USD 顺序使用 NPZ 的局部平移和 SDK 四元数，设置单位缩放。写入 USD 四元数时将标量分量 `w` 放在 `Gf.Quatf` 的首个参数。

重新导出需安装 OpenUSD 的 Python 绑定。给 `python tools/export_friston_assets.py` 传入 G-i Toolkit 参考检出目录。加 `--check` 可逐字节核对现有资产。
