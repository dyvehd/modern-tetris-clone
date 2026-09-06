"""Scoring, attack (garbage sent), back-to-back and combo rules.

Tables follow Jstris, per https://harddrop.com/wiki/Jstris:

- Attack: Single 0, Double 1, Triple 2, Quad 4, TSS 2, TSD 4, TST 6,
  Mini T-spin Single 0 (keeps the B2B chain but earns no B2B bonus),
  Mini T-spin Double counts as a TSD. B2B: +1. Perfect Clear: 10 (replaces
  the base attack). Combo table below, 12+ capped at +5.
- Score: Single 100, Double 300, Triple 500, Quad 800, T-spin 400/800/1200/1600,
  Mini T-spin 100 / Mini TSS 200, Perfect Clear +3000 (added),
  B2B: +50% on difficult clears (mini TSS excluded), combos +(50 * combo).
  Soft drop 1/cell, hard drop 2/cell (handled by the game, not here).

Combo counting: ``combo`` is the number of consecutive line-clearing locks
(first clear = 1). The wiki's combo table is 0-indexed ("combo 0" is the
first clear), so the attack bonus for consecutive-clear count ``combo`` is
``COMBO_ATTACK[combo - 1]``: the 3rd consecutive clear is the first to add
garbage (+1). Combo score bonus is 50 * (combo - 1), i.e. +50 from the 2nd
consecutive clear on.
"""

from __future__ import annotations

from dataclasses import dataclass

# kind: "none" (plain clear) | "full" (T-spin) | "mini" (mini T-spin)
BASE_SCORE: dict[tuple[int, str], int] = {
    (1, "none"): 100,
    (2, "none"): 300,
    (3, "none"): 500,
    (4, "none"): 800,
    (0, "full"): 400,
    (1, "full"): 800,
    (2, "full"): 1200,
    (3, "full"): 1600,
    (0, "mini"): 100,
    (1, "mini"): 200,
}

# Attack for a difficult clear that extends an existing B2B chain.
B2B_ATTACK_BONUS = 1
B2B_SCORE_MULTIPLIER = 1.5
PERFECT_CLEAR_ATTACK = 10
PERFECT_CLEAR_SCORE = 3000
COMBO_SCORE_STEP = 50

# Indexed by (consecutive clears - 1); index 12 and above -> 5.
COMBO_ATTACK: tuple[int, ...] = (0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5)


def combo_attack_bonus(combo: int) -> int:
    if combo <= 1:
        return 0
    return COMBO_ATTACK[min(combo - 1, len(COMBO_ATTACK) - 1)]


@dataclass
class ClearOutcome:
    lines: int
    kind: str  # "none" | "full" | "mini"
    difficult: bool  # extends/maintains the B2B chain
    perfect_clear: bool
    score: int  # points awarded by this clear (before drop points)
    attack: int  # garbage lines sent (after B2B/combo/PC modifiers)

    @property
    def label(self) -> str:
        names = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "QUAD"}
        if self.kind == "full":
            base = {0: "T-SPIN", 1: "T-SPIN SINGLE", 2: "T-SPIN DOUBLE", 3: "T-SPIN TRIPLE"}
        elif self.kind == "mini":
            base = {0: "T-SPIN MINI", 1: "T-SPIN MINI SINGLE"}
            if self.lines >= 2:  # mini doubles count as full TSDs (Jstris)
                base = {2: "T-SPIN DOUBLE", 3: "T-SPIN TRIPLE"}
        else:
            base = {0: "", **names}
        text = base.get(self.lines, "QUAD" if self.lines > 4 else "")
        if self.perfect_clear:
            text += " PERFECT CLEAR"
        if self.difficult and self.b2b_extended:
            text = "B2B " + text
        return text.strip()

    b2b_extended: bool = False  # whether the B2B bonus applied


def evaluate_clear(
    lines: int,
    kind: str,
    combo: int,  # consecutive-clear count INCLUDING this clear (>=1 when lines>0)
    b2b_chain_before: int,  # chain length before this clear
    perfect_clear: bool,
    level_multiplier: int = 1,  # Guideline marathon style; Jstris uses 1
) -> ClearOutcome:
    """Compute score/attack for one line-clearing (or T-spin 0-line) lock."""
    # Mini T-spin doubles/triples count as full T-spins (Jstris rule).
    if kind == "mini" and lines >= 2:
        kind = "full"

    difficult = lines > 0 and (lines == 4 or kind in ("full", "mini"))

    base = BASE_SCORE.get((lines, kind), 0) if (lines > 0 or kind != "none") else 0

    # B2B bonus: difficult clear that extends an existing chain. A mini T-spin
    # single keeps the chain but earns no bonus (Jstris note).
    b2b_extended = bool(
        difficult and b2b_chain_before >= 1 and not (kind == "mini" and lines == 1)
    )
    score = int(base * B2B_SCORE_MULTIPLIER) * level_multiplier if b2b_extended else base * level_multiplier

    if lines <= 0:
        attack = 0
    elif kind == "none":
        attack = {1: 0, 2: 1, 3: 2, 4: 4}.get(lines, 4)
    elif kind == "mini":
        attack = 0  # mini T-spin single sends nothing
    else:
        attack = {1: 2, 2: 4, 3: 6}.get(lines, 6)

    if perfect_clear:
        attack = PERFECT_CLEAR_ATTACK
        score += PERFECT_CLEAR_SCORE * level_multiplier
    if b2b_extended:
        attack += B2B_ATTACK_BONUS
    if combo > 1:
        score += COMBO_SCORE_STEP * (combo - 1) * level_multiplier
    if lines > 0:
        attack += combo_attack_bonus(combo)

    return ClearOutcome(
        lines=lines,
        kind=kind,
        difficult=difficult,
        perfect_clear=perfect_clear,
        score=score,
        attack=attack,
        b2b_extended=b2b_extended,
    )
