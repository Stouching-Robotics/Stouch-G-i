from enum import Enum


class MoCapHandStateEnum(str, Enum):
    UNCONNECTED = "unconnected"
    CONNECTED = "connected"
    EXITED = "exited"


class MoCapCalibrateRootType(str, Enum):
    HORIZONTAL = "horizontal"      # 基准：手掌摊平、手背朝上
    VERTICAL_UP = "vertical_up"    # 绕横掌轴（MANO Z）抬起 90°
    VERTICAL_DOWN = "vertical_down"  # 绕横掌轴俯下 90°
    ROLL = "roll"                  # 绕指尖轴（MANO X）翻掌 90°
    YAW = "yaw"                    # 绕竖直轴（MANO Y）水平转 90°
    RESET = "reset"  #
    CALC = "calc"  #


class MoCapCalibrateInstallationType(str, Enum):
    POSE_0 = "pose_0"  # 五指张开
    POSE_2 = "pose_2"  # 四指合拢
    POSE_3 = "pose_3"  # 五指并拢触碰拇指
    RESET = "reset"  #
    CALC = "calc"  #


class MoCapCalibrateShapeType(str, Enum):
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"
    RESET = "reset"
    CALC = "calc"  #


class MoCapCalibrateFistType(str, Enum):
    """四指握拳标定：实测四指弯曲时的横向偏移（思路一），供运行时减去。"""
    CAPTURE = "capture"  # 四指握拳采集弯曲位姿
    RESET = "reset"
    CALC = "calc"


class MoCapCalibrateShapeKeyPointType(tuple[str, int], Enum):
    T_MCP = ("T_MCP", 0)
    T_PIP = ("T_PIP", 1)
    T_DIP = ("T_DIP", 2)
    T_TIP = ("T_TIP", 3)
    I_MCP = ("I_MCP", 4)
    I_PIP = ("I_PIP", 5)
    I_DIP = ("I_DIP", 6)
    I_TIP = ("I_TIP", 7)
    M_MCP = ("M_MCP", 8)
    M_PIP = ("M_PIP", 9)
    M_DIP = ("M_DIP", 10)
    M_TIP = ("M_TIP", 11)
    R_MCP = ("R_MCP", 12)
    R_PIP = ("R_PIP", 13)
    R_DIP = ("R_DIP", 14)
    R_TIP = ("R_TIP", 15)
    L_MCP = ("L_MCP", 16)
    L_PIP = ("L_PIP", 17)
    L_DIP = ("L_DIP", 18)
    L_TIP = ("L_TIP", 19)
    W_LEFT = ("W_LEFT", 20)
    W_RIGHT = ("W_RIGHT", 21)
    RESET = ("reset", None)
    CALC = ("calc", None)



class MoCapCalibrateWristType(str, Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"
    RESET = "reset"
    CALC = "calc"


class MoCapHandIDEnum(str, Enum):
    UNKNOWN = 'unknown'
    NONE = 'none'
    LEFT = 'left'
    RIGHT = 'right'
    SPIKE = 'spike'
