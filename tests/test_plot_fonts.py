from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
import os
import tempfile
import tomllib

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pred_test_matplotlib")
)

import matplotlib.pyplot as plt
from fontTools.ttLib import TTFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_fonts import (  # noqa: E402
    CHINESE_FONT_PATH,
    WEB_FONT_FAMILY,
    register_matplotlib_chinese_font,
)


class PlotFontTests(unittest.TestCase):
    def test_bundled_font_exists_and_covers_required_chinese(self):
        self.assertTrue(CHINESE_FONT_PATH.is_file())
        font = TTFont(CHINESE_FONT_PATH, lazy=True)
        codepoints = {
            codepoint
            for table in font["cmap"].tables
            for codepoint in table.cmap
        }
        for character in "实验收益开仓胜率盈亏累计分布正态":
            self.assertIn(ord(character), codepoints)
        font.close()

    def test_streamlit_theme_uses_static_bundled_font_without_global_css(self):
        config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
        self.assertTrue(config["server"]["enableStaticServing"])
        self.assertIn(WEB_FONT_FAMILY, config["theme"]["font"])
        self.assertIn(WEB_FONT_FAMILY, config["theme"]["headingFont"])
        font_face = config["theme"]["fontFaces"][0]
        self.assertEqual(font_face["family"], WEB_FONT_FAMILY)
        self.assertEqual(
            font_face["url"], f"app/static/fonts/{CHINESE_FONT_PATH.name}"
        )
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertNotIn("web_font_css", source)
        self.assertNotIn("unsafe_allow_html=True", source)

    def test_matplotlib_uses_bundled_font_without_missing_glyph_warning(self):
        family = register_matplotlib_chinese_font()
        self.assertEqual(family, "Noto Sans CJK SC")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            figure, axis = plt.subplots()
            axis.set_title("实验收益开仓胜率盈亏累计分布正态")
            figure.canvas.draw()
            plt.close(figure)
        self.assertFalse(any("Glyph" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
