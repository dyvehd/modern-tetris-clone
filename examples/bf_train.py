"""Round-2 value training — V on blockfish contrast labels.

Same recipe as round 1 (examples/value_experiment.py stage_train):
twin-headed ValueNet, weighted MSE + BCE, packed arrays, contiguous
held-out tail (fresh seeds). New for round 2 — the diagnostics that
answer the round-1 diagnosis directly:

- within-board rank accuracy: on held-out ROOT decision blocks, how
  often does q̂ rank the true best (min-q) candidate first (chance =
  1/n), and the mean Spearman-ish concordance of (q̂, q) orderings
  (Kendall-tau-like: fraction of concordant candidate pairs).
- root q̂ spread: mean (max q̂ - min q̂) per root — round 1 measured
  ~2 pieces of meaningless spread; contrast labels should make the
  spread real (aligned with label spread).

Run (server):  python3 examples/bf_train.py --epochs 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tetris.ai.value import (  # noqa: E402
    ValueNet,
    ValueTrainer,
    evaluate_q_mae,
    load_value,
    save_value,
)

MODELS = Path("models/value")
SOURCES = ["L1", "L3", "L5", "L10"]


def _load_packed():
    xs, qs, fs, es, irs = [], [], [], [], []
    for name in SOURCES:
        z = np.load(MODELS / f"round2_{name}.npz")
        xs.append(z["x"])
        qs.append(z["q"])
        fs.append(z["failed"])
        es.append(z["explored"])
        irs.append(z["is_root"])
        z.close()
    return (
        np.concatenate(xs),
        np.concatenate(qs),
        np.concatenate(fs),
        np.concatenate(es),
        np.concatenate(irs),
    )


def _root_segments(is_root):
    """Row index ranges of each root decision block (consecutive
    is_root runs)."""
    rows = np.where(is_root)[0]
    if len(rows) == 0:
        return []
    breaks = np.where(np.diff(rows) > 1)[0] + 1
    return np.split(rows, breaks)


def root_diagnostics(net, x, q, is_root, device: str) -> dict:
    """Held-out within-board discrimination on root blocks."""
    net.eval()
    segs = _root_segments(is_root)
    if not segs:
        return {}
    q_hats = np.empty(len(q), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(q), 65536):
            xt = torch.as_tensor(x[i : i + 65536], dtype=torch.float32, device=device)
            qh, _ = net(xt)
            q_hats[i : i + 65536] = qh.cpu().numpy()
    best_first = 0
    concordant = 0
    pairs = 0
    spreads = []
    label_spreads = []
    for s in segs:
        if len(s) < 3:
            continue
        qs_hat = q_hats[s]
        qs_lab = q[s]
        best_first += int(np.argmin(qs_hat) == np.argmin(qs_lab))
        order_hat = np.argsort(qs_hat)
        order_lab = np.argsort(qs_lab)
        # concordant fraction over all candidate pairs (Kendall-like)
        n = len(s)
        for a in range(n):
            for b in range(a + 1, n):
                pairs += 1
                sh = qs_hat[a] < qs_hat[b]
                sl = qs_lab[a] < qs_lab[b]
                if sh == sl:
                    concordant += 1
        spreads.append(float(qs_hat.max() - qs_hat.min()))
        label_spreads.append(float(qs_lab.max() - qs_lab.min()))
    return {
        "root_decisions": len(segs),
        "best_first_rate": best_first / max(1, len(segs)),
        "pair_concordance": concordant / max(1, pairs),
        "mean_qhat_spread": float(np.mean(spreads)) if spreads else None,
        "mean_label_spread": float(np.mean(label_spreads)) if label_spreads else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    x, q, failed, explored, is_root = _load_packed()
    n = len(q)
    print(f"loaded {n} rows from {len(SOURCES)} sources on {device}", flush=True)

    n_eval = max(1, n // 10)
    eval_packed = (x[-n_eval:], q[-n_eval:], failed[-n_eval:], explored[-n_eval:])
    eval_root = is_root[-n_eval:]
    train_packed = (x[:-n_eval], q[:-n_eval], failed[:-n_eval], explored[:-n_eval])

    net = ValueNet(hidden=args.hidden, layers=args.layers, seed=args.seed)
    trainer = ValueTrainer(
        net, lr=args.lr, device=device, chunk_rows=args.chunk,
        rng=np.random.default_rng(args.seed),
    )
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        stats = trainer.train_epoch(train_packed)
        if epoch == 1 or epoch % 5 == 0:
            mae = evaluate_q_mae(net, eval_packed, device)
            diag = root_diagnostics(net, *eval_packed[:2], eval_root, device)
            marker = ""
            if mae < best_mae:
                best_mae = mae
                save_value(net, MODELS / "value_round2.json")
                marker = " *saved*"
            print(
                f"epoch {epoch}/{args.epochs} | loss {stats.loss:.4f} "
                f"(q {stats.q_loss:.4f}, fail {stats.fail_loss:.4f}) "
                f"| held-out q-MAE {mae:.3f} "
                f"| root best-first {diag.get('best_first_rate', float('nan')):.3f} "
                f"pair-concord {diag.get('pair_concordance', float('nan')):.3f} "
                f"| q̂ spread {diag.get('mean_qhat_spread', float('nan')):.2f} "
                f"vs label {diag.get('mean_label_spread', float('nan')):.2f}"
                f" | {stats.seconds:.1f}s{marker}",
                flush=True,
            )
    print(f"best held-out q-MAE {best_mae:.3f}; saved {MODELS}/value_round2.json", flush=True)


if __name__ == "__main__":
    main()
