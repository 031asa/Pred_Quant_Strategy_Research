from __future__ import annotations

"""项目自带中文字体在浏览器Plotly和服务器Matplotlib中的统一配置。"""

from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHINESE_FONT_PATH = PROJECT_ROOT / "static" / "fonts" / "NotoSansCJKsc-Regular.otf"
CHINESE_FONT_FAMILY = "Noto Sans CJK SC"
WEB_FONT_FAMILY = "Pred Noto Sans CJK SC"
PLOTLY_FONT_STACK = (
    f"{WEB_FONT_FAMILY}, {CHINESE_FONT_FAMILY}, Microsoft YaHei, "
    "PingFang SC, SimHei, sans-serif"
)


@lru_cache(maxsize=1)
def register_matplotlib_chinese_font() -> str:
    """运行时注册项目字体，不依赖Windows或Linux系统字体目录。"""

    if not CHINESE_FONT_PATH.is_file():
        raise FileNotFoundError(f"项目中文字体不存在：{CHINESE_FONT_PATH}")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
    resolved_name = font_manager.FontProperties(
        fname=str(CHINESE_FONT_PATH)
    ).get_name()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        resolved_name,
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return resolved_name
