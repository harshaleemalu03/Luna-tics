import sys
import base64
import os

if len(sys.argv) < 3:
    print("Usage: python write_code.py <filepath> <base64_content>")
    sys.exit(1)

filepath = sys.argv[1]
b64_str = sys.argv[2]
content = base64.b64decode(b64_str).decode("utf-8")

os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Successfully wrote {filepath} ({len(content)} chars)")
