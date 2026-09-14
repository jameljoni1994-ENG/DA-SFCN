from pathlib import Path
import re
import sys
from fontTools.ttLib import TTFont

out = Path(r"C:\Users\Windows.11\Desktop\phd\results\glyph_report.txt")
font_path = Path(r"C:\Users\Windows.11\AppData\Local\Programs\MiKTeX\fonts\truetype\public\amiri\Amiri-Regular.ttf")
font = TTFont(str(font_path))
cmap = {}
for t in font["cmap"].tables:
    cmap.update(t.cmap)

root = Path(r"C:\Users\Windows.11\Desktop\phd\papers")
files = list(root.glob("ar/*.tex")) + list(root.glob("unified/paper_ar.tex"))
missing = {}
cyrillic_hits = []
lines_out = []

for tex in files:
    text = tex.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        for ch in line:
            o = ord(ch)
            if o < 128:
                continue
            if 0x0400 <= o <= 0x04FF:
                cyrillic_hits.append((tex.name, i, ch, f"U+{o:04X}", line.strip()[:140]))
            if ch not in cmap:
                missing.setdefault((o, ch), set()).add(f"{tex.name}:{i}")

lines_out.append(f"cmap size {len(cmap)}")
lines_out.append(f"missing unique chars {len(missing)}")
for (o, ch), locs in sorted(missing.items()):
    lines_out.append(f"U+{o:04X} {ch!r} -> {sorted(locs)[:5]}")

lines_out.append("\n=== CYRILLIC CONTAMINATION ===")
for h in cyrillic_hits[:50]:
    lines_out.append(f"{h[0]}:{h[1]} {h[3]} {h[2]!r} :: {h[4]}")

# secular word codepoints
lines_out.append("\n=== سكانت codepoints ===")
for tex in files:
    text = tex.read_text(encoding="utf-8")
    for m in re.finditer(r".{0,8}سكا.{0,12}", text):
        s = m.group(0)
        cps = " ".join(f"{c}=U+{ord(c):04X}" for c in s)
        lines_out.append(s)
        lines_out.append(cps)

# contributions line with boxes
for tex in files:
    text = tex.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        if "السكانت" in line or "سكانت" in line or "التعامد" in line:
            lines_out.append(f"\nLINE {tex.name}:{i}")
            lines_out.append(line)
            lines_out.append(" ".join(f"{c}=U+{ord(c):04X}" for c in line if ord(c) > 127))

out.write_text("\n".join(lines_out), encoding="utf-8")
print("wrote", out)
print("missing", len(missing), "cyrillic hits", len(cyrillic_hits))
