import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# 五指张开（FIVE）
mano_model_pose_0 = np.zeros(shape=(48,))

trivial_mano_pose = R.from_rotvec(np.zeros(shape=(16, 3)))
trivial_mano_tensor = torch.zeros(16, 3)
trivial_position = np.zeros(shape=(3,))
