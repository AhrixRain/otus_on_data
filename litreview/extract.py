import os, sys, glob
from pypdf import PdfReader

src = r"C:\Users\AhrixMarin\Desktop\otus\litreview\pdfs"
dst = r"C:\Users\AhrixMarin\Desktop\otus\litreview\txt"
os.makedirs(dst, exist_ok=True)

for pdf in sorted(glob.glob(os.path.join(src, "*.pdf"))):
    base = os.path.splitext(os.path.basename(pdf))[0]
    out = os.path.join(dst, base + ".txt")
    try:
        r = PdfReader(pdf)
        parts = []
        for i, p in enumerate(r.pages):
            try:
                parts.append(p.extract_text() or "")
            except Exception as e:
                parts.append(f"\n[[page {i+1} extract error: {e}]]\n")
        txt = "\n\n=== PAGE BREAK ===\n\n".join(parts)
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"{base}: {len(r.pages)} pages, {len(txt)} chars -> {out}")
    except Exception as e:
        print(f"{base}: FAILED {e}")
