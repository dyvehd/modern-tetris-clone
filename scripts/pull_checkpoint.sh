#!/usr/bin/env bash
# Pull a checkpoint from the Molab notebook through the console, in
# gzip+base85 chunks, verify md5, stage locally. Usage:
#   MOLAB_URL=... MOLAB_TOKEN=... ./pull_checkpoint.sh <name> <out_path>
# URL/TOKEN are read from the environment (never hard-code a token here —
# see the security note in docs/onboarding-guide.md).
set -euo pipefail
URL="${MOLAB_URL:?set MOLAB_URL to the notebook base URL}"
TOKEN="${MOLAB_TOKEN:?set MOLAB_TOKEN to the notebook auth token}"
NAME="$1"; OUT="$2"
EXEC="/home/dathuynh/.zcode/skills/marimo-pair/scripts/execute-code.sh"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# manifest: how many chunks + expected md5s (fresh snapshot first — the
# runner rewrites files mid-run, so all chunks must come from ONE snapshot)
SNAP=$("$EXEC" --url "$URL" --token "$TOKEN" - <<PY 2>/dev/null | grep -oE "SNAPTAG [0-9]+" | cut -d' ' -f2
import sys, importlib
sys.path.insert(0, "/marimo")
import pull_checkpoint as pc
importlib.reload(pc)
print("SNAPTAG", pc.snapshot())
PY
)
MANIFEST=$("$EXEC" --url "$URL" --token "$TOKEN" - <<PY
import sys, json, importlib
sys.path.insert(0, "/marimo")
import pull_checkpoint as pc
importlib.reload(pc)
print("@@MAN@@" + json.dumps(pc.manifest("$SNAP")["$NAME"]))
PY
)
META=$(grep -o '@@MAN@@.*' <<<"$MANIFEST" | sed 's/@@MAN@@//')
CHUNKS=$(python3 -c "import json; print(json.loads('''$META''')['chunks'])")
GZMD5=$(python3 -c "import json; print(json.loads('''$META''')['gz_md5'])")
RAWMD5=$(python3 -c "import json; print(json.loads('''$META''')['raw_md5'])")
echo "pulling $NAME: $CHUNKS chunk(s), raw-md5 $RAWMD5"

for ((k=0; k<CHUNKS; k++)); do
  "$EXEC" --url "$URL" --token "$TOKEN" - > "$TMP/chunk$k.log" 2>/dev/null <<PY
import sys, importlib
sys.path.insert(0, "/marimo")
import pull_checkpoint as pc
importlib.reload(pc)
pc.serve("$NAME", $k, tag="$SNAP")
PY
  python3 - "$TMP/chunk$k.log" "$TMP/part$k.b85" <<'PY'
import sys
# frame: CHK.<k>.<n>.<len>.<payload>.END  (dots cannot appear in base85)
# the log may carry a connection-warning line first — take the CHK line
line = next(l for l in open(sys.argv[1]) if l.startswith("CHK."))
fields = line.strip().split(".")
assert fields[0] == "CHK" and fields[-1] == "END" and len(fields) == 6, \
    f"bad chunk frame: {len(fields)} fields"
k, n, ln, payload = int(fields[1]), int(fields[2]), int(fields[3]), fields[4]
assert len(payload) == ln, (k, len(payload), ln)
with open(sys.argv[2], "w") as f:
    f.write(payload)
PY
done

cat "$TMP"/part*.b85 > "$TMP/all.b85"
python3 - "$TMP/all.b85" "$GZMD5" "$RAWMD5" "$OUT" <<'PY'
import base64, gzip, hashlib, sys
b85 = open(sys.argv[1]).read()
gz = base64.b85decode(b85)          # wire bytes: must match the frozen .gz
assert hashlib.md5(gz).hexdigest() == sys.argv[2], "gz md5 mismatch (chunk corrupted)"
raw = gzip.decompress(gz)           # gunzip: must match the checkpoint itself
assert hashlib.md5(raw).hexdigest() == sys.argv[3], "raw md5 mismatch"
open(sys.argv[4], "wb").write(raw)
print("OK: md5 verified,", len(raw), "bytes ->", sys.argv[4])
PY
