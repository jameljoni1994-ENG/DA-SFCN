"""Sanitize Arabic TeX sources: remove Cyrillic lookalikes and bad punctuation."""
from pathlib import Path
import re

# Cyrillic letters that often sneak in as Latin/Arabic lookalikes
CYR_TO_AR = {
    "\u0430": "\u0627",  # а -> ا
    "\u0410": "\u0627",  # А -> ا
    "\u0435": "\u0647",  # е -> ه (approx) — better map carefully
    "\u043A": "\u0643",  # к -> ك
    "\u041A": "\u0643",  # К -> ك
    "\u043D": "\u0646",  # н -> ن
    "\u041D": "\u0646",  # Н -> ن
    "\u043E": "\u0648",  # о -> و (weak)
    "\u0440": "\u0631",  # р -> ر
    "\u0441": "\u0633",  # с -> س
    "\u0443": "\u0648",  # у
    "\u0445": "\u062E",  # х -> خ
    "\u0443": "\u0648",
}

# Safer: only map the ones we confirmed, and flag others
SAFE_MAP = {
    "\u0430": "ا",
    "\u0410": "ا",
    "\u043A": "ك",
    "\u041A": "ك",
    "\u043D": "ن",
    "\u041D": "ن",
    "\u00AB": '"',
    "\u00BB": '"',
}

root = Path(r"C:\Users\Windows.11\Desktop\phd\papers")
files = list(root.glob("ar/*.tex")) + [root / "unified" / "paper_ar.tex"]
report = []

for tex in files:
    text = tex.read_text(encoding="utf-8")
    orig = text
    for bad, good in SAFE_MAP.items():
        if bad in text:
            count = text.count(bad)
            text = text.replace(bad, good)
            report.append(f"{tex.name}: replaced U+{ord(bad):04X} x{count} -> {good!r}")
    # remaining Cyrillic
    left = sorted({c for c in text if 0x0400 <= ord(c) <= 0x04FF})
    if left:
        report.append(f"{tex.name}: REMAINING CYRILLIC {[f'U+{ord(c):04X}' for c in left]}")
    if text != orig:
        tex.write_text(text, encoding="utf-8")
        report.append(f"{tex.name}: WRITTEN")

out = Path(r"C:\Users\Windows.11\Desktop\phd\results\sanitize_report.txt")
out.write_text("\n".join(report) if report else "nothing to fix", encoding="utf-8")
print("\n".join(report) if report else "nothing")
