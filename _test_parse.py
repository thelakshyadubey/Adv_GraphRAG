import time
from hybrid_rag.ingestion.parser import parse

t0 = time.time()
doc = parse(r"C:\Users\Lakshya\Desktop\TEST\2.pdf")
elapsed = round(time.time() - t0, 1)

print(f"Time       : {elapsed}s")
print(f"Pages      : {len(doc.pages)}")
print(f"OCR pages  : {doc.metadata.get('ocr_pages', 0)}")
print(f"Total chars: {len(doc.text)}")
print()
print("=== PAGE 1 (first 1000 chars) ===")
print(doc.pages[0][:1000] if doc.pages else "(empty)")
print()
print("=== PAGE 2 (first 500 chars) ===")
print(doc.pages[1][:500] if len(doc.pages) > 1 else "(only 1 page)")
