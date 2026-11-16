#!/usr/bin/env bash
# Build the JMLR manuscript with tectonic.
set -euo pipefail
cd "$(dirname "$0")"
tectonic -X compile main.tex
ls -la main.pdf
python3 - <<'EOF'
import re
try:
    from pypdf import PdfReader
    n = len(PdfReader('main.pdf').pages)
    print(f'main.pdf: {n} pages (JMLR prefers <= 35, needs a justification above 50)')
except ImportError:
    print('pypdf not installed: count the pages by hand')
tex = open('main.tex').read()
m = re.search(r'\\begin\{abstract\}%?(.*?)\\end\{abstract\}', tex, re.S)
words = len(m.group(1).split())
print(f'abstract: {words} words (JMLR limit 200)')
EOF
