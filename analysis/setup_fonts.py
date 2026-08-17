"""Chinese font setup for matplotlib.

Fixes the Chinese character rendering issue (garbled squares/missing glyphs) by
configuring matplotlib's font.sans-serif list to prefer WenQuanYi Zen Hei (文泉驿正黑)
which is available on this system via /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc.

Usage:
    from analysis.setup_fonts import setup_chinese_font
    setup_chinese_font()   # call BEFORE importing pyplot

Or run as a module and it will auto-apply.
"""
import matplotlib
import matplotlib.font_manager as fm
import os


# 中文字体优先级（Linux 系统）
CHINESE_FONTS = [
    'WenQuanYi Zen Hei',     # 文泉驿正黑（Linux 默认）
    'Noto Sans CJK SC',       # Google Noto（如果安装）
    'Source Han Sans CN',     # 思源黑体
    'PingFang SC',            # macOS
    'Microsoft YaHei',        # Windows
    'SimHei',                 # Windows 简体
    'STHeiti',                # macOS 简体
    'Hiragino Sans GB',       # macOS 简体
    'DejaVu Sans',            # fallback
]


def _find_chinese_font() -> str:
    """Detect the first available Chinese font on this system."""
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in CHINESE_FONTS:
        if font_name in available:
            return font_name
    return 'DejaVu Sans'


def setup_chinese_font():
    """Configure matplotlib for proper Chinese rendering.

    Idempotent — safe to call multiple times.
    """
    font = _find_chinese_font()

    # 强制重建字体缓存（确保新装的字体被识别）
    cache_dir = matplotlib.get_cachedir()
    for cache_file in ['fontlist-v330.json', 'fontlist-v390.json',
                       'fontlist-v400.json', 'fontlist-v410.json']:
        cache_path = os.path.join(cache_dir, cache_file)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass
    # 重建字体管理器
    fm._load_fontmanager(try_read_cache=False)

    # 设置 matplotlib rcParams
    matplotlib.rcParams['font.sans-serif'] = [font] + CHINESE_FONTS
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

    # 验证
    available = {f.name for f in fm.fontManager.ttflist}
    verified = font in available
    return font, verified


if __name__ == '__main__':
    font, ok = setup_chinese_font()
    print(f"Chinese font: {font}")
    print(f"Verified available: {ok}")
    print(f"matplotlib rcParams['font.sans-serif']: {matplotlib.rcParams['font.sans-serif']}")
    print(f"matplotlib rcParams['axes.unicode_minus']: {matplotlib.rcParams['axes.unicode_minus']}")