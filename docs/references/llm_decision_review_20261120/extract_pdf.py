"""Extract only PDFs retained in this literature-review directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WHEEL = ROOT.parent / "task_contrast_20261120/pypdf-5.9.0-py3-none-any.whl"
sys.path.insert(0, str(WHEEL))
from pypdf import PdfReader

for original in sorted(ROOT.glob("*.pdf")):
    reader = PdfReader(original)
    text = "\n".join(
        f"\n=== PDF PAGE {i} ===\n" + page.extract_text()
        for i, page in enumerate(reader.pages, 1)
    )
    original.with_suffix(".txt").write_text(text)
    print(original.name, len(reader.pages), "pages")
