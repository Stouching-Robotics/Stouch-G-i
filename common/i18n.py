"""界面语言切换。

用法就一个函数:

    from common.i18n import L
    psim.Text(L("未完成", "Not done"))

为什么用「两个字面量并列」而不是 key→查表:
  - 调用处直接看得到两种文案，不用跳到别处对照，也不会出现"key 改了忘了改表"
  - f-string 插值天然可用: L(f"剩 {n} 帧", f"{n} frames left")
  - 没有 key 命名负担，新增文案零成本

窗口标题要用 wid() 包一层。ImGui 用「可见标题」当窗口 ID，标题一变就会被当成
一个全新窗口（位置大小全部重置）；"标题##固定id" 的写法能让 ID 保持不变，
`##` 之后的部分不显示。
"""

from typing import Literal

Lang = Literal["zh", "en"]

_LANG: Lang = "zh"


def set_lang(lang: Lang):
    global _LANG
    _LANG = "en" if str(lang).lower().startswith("en") else "zh"


def get_lang() -> Lang:
    return _LANG


def is_en() -> bool:
    return _LANG == "en"


def L(zh: str, en: str) -> str:
    """按当前语言二选一。"""
    return en if _LANG == "en" else zh


def wid(zh: str, en: str, ident: str) -> str:
    """带固定 ID 的窗口标题 / 控件标签，切语言时不会丢失窗口位置和控件状态。"""
    return f"{L(zh, en)}##{ident}"
