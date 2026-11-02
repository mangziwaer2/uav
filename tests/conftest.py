"""测试只导入当前src包，不扫描历史归档。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
