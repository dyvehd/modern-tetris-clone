#!/usr/bin/env python
"""Pull value-net artifacts from the Molab notebook console.

Same wire protocol as scripts/pull_checkpoint.sh (gzip + base85 chunks,
dual md5), but driven from Python so quoting is trivial. Credentials
come from the environment (MOLAB_URL / MOLAB_TOKEN) — never committed.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import subprocess
import sys

EXEC = "/home/dathuynh/.zcode/skills/marimo-pair/scripts/execute-code.sh"


def call(code: str) -> str:
    r = subprocess.run(
        [EXEC, "--url", os.environ["MOLAB_URL"],
         "--token", os.environ["MOLAB_TOKEN"], "-c", code],
        capture_output=True, text=True, timeout=180,
    )
    out = r.stdout
    # strip the non-local warning line if present
    lines = [l for l in out.splitlines() if not l.startswith("Warning:")]
    return "\n".join(lines)


def pull(name: str, out_path: str) -> None:
    meta_s = call(
        "import sys, json; sys.path.insert(0, '/marimo'); "
        "import pull_checkpoint as pc; "
        "print('@@MAN@@' + json.dumps(pc.manifest()['%s']))" % name
    )
    meta = json.loads(meta_s.split("@@MAN@@", 1)[1].strip().splitlines()[0])
    gz_md5, raw_md5, chunks = meta["gz_md5"], meta["raw_md5"], meta["chunks"]
    print(f"pulling {name}: {chunks} chunk(s), raw-md5 {raw_md5}")
    parts = []
    for k in range(chunks):
        line = call(
            "import sys; sys.path.insert(0, '/marimo'); "
            "import pull_checkpoint as pc; pc.serve('%s', %d)" % (name, k)
        )
        line = next(l for l in line.splitlines() if l.startswith("CHK."))
        fields = line.strip().split(".")
        assert fields[0] == "CHK" and fields[-1] == "END", "bad frame"
        # frame: CHK.<k>.<n>.<len>.<payload>.END — payload may not contain '.'
        assert len(fields) == 6, f"unexpected fields {len(fields)}"
        payload = fields[4]
        assert len(payload) == int(fields[3]), "length mismatch"
        parts.append(payload)
    gz = base64.b85decode("".join(parts))
    assert hashlib.md5(gz).hexdigest() == gz_md5, "gz md5 mismatch"
    raw = gzip.decompress(gz)
    assert hashlib.md5(raw).hexdigest() == raw_md5, "raw md5 mismatch"
    with open(out_path, "wb") as f:
        f.write(raw)
    print(f"OK: {len(raw)} bytes -> {out_path} (md5 verified)")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "models/value"
    os.makedirs(out_dir, exist_ok=True)
    for name in (
        "round1_manifest.json", "value_round1_eval.json",
        "value_round1.json", "value_round1b.json",
        # round 2 (blockfish labels)
        "blockfish_manifest.json", "value_round2.json",
        "value_round2_eval.json",
    ):
        pull(name, os.path.join(out_dir, name))
