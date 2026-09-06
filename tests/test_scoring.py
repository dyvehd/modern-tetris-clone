"""Jstris scoring / attack / B2B / combo / perfect clear rules.
Reference: https://harddrop.com/wiki/Jstris"""

from tetris.engine.scoring import combo_attack_bonus, evaluate_clear


def test_base_scores():
    assert evaluate_clear(1, "none", 1, 0, False).score == 100
    assert evaluate_clear(2, "none", 1, 0, False).score == 300
    assert evaluate_clear(3, "none", 1, 0, False).score == 500
    assert evaluate_clear(4, "none", 1, 0, False).score == 800


def test_tspin_scores():
    assert evaluate_clear(0, "full", 0, 0, False).score == 400
    assert evaluate_clear(1, "full", 1, 0, False).score == 800
    assert evaluate_clear(2, "full", 1, 0, False).score == 1200
    assert evaluate_clear(3, "full", 1, 0, False).score == 1600
    assert evaluate_clear(0, "mini", 0, 0, False).score == 100
    assert evaluate_clear(1, "mini", 1, 0, False).score == 200


def test_tspin_attack():
    assert evaluate_clear(1, "full", 1, 0, False).attack == 2
    assert evaluate_clear(2, "full", 1, 0, False).attack == 4
    assert evaluate_clear(3, "full", 1, 0, False).attack == 6
    assert evaluate_clear(0, "full", 0, 0, False).attack == 0


def test_mini_tspin_single_sends_nothing_but_keeps_chain():
    r = evaluate_clear(1, "mini", 1, 0, False)
    assert r.attack == 0
    assert r.difficult is True

    # with an active chain: keeps it, but no +1 bonus
    r2 = evaluate_clear(1, "mini", 1, 3, False)
    assert r2.attack == 0
    assert r2.b2b_extended is False


def test_line_clear_attack():
    assert evaluate_clear(1, "none", 1, 0, False).attack == 0
    assert evaluate_clear(2, "none", 1, 0, False).attack == 1
    assert evaluate_clear(3, "none", 1, 0, False).attack == 2
    assert evaluate_clear(4, "none", 1, 0, False).attack == 4


def test_b2b_chain_bonus():
    first = evaluate_clear(4, "none", 1, 0, False)  # quad, no chain yet
    assert first.attack == 4
    assert first.b2b_extended is False

    second = evaluate_clear(4, "none", 1, 1, False)  # quad in b2b
    assert second.attack == 5
    assert second.score == 1200  # 800 * 1.5

    tsd_in_b2b = evaluate_clear(2, "full", 1, 1, False)
    assert tsd_in_b2b.attack == 5
    assert tsd_in_b2b.score == 1800


def test_single_breaks_chain():
    r = evaluate_clear(1, "none", 1, 4, False)
    assert r.difficult is False
    assert r.b2b_extended is False


def test_zero_line_locks_do_not_touch_chain():
    r = evaluate_clear(0, "full", 0, 2, False)
    assert r.difficult is False  # doesn't extend...
    # (the game keeps the chain value itself for non-clearing locks)


def test_combo_attack_table():
    # Jstris table, 0-indexed: first two consecutive clears add nothing.
    assert [combo_attack_bonus(c) for c in range(1, 8)] == [0, 0, 1, 1, 1, 2, 2]
    assert combo_attack_bonus(13) == 5
    assert combo_attack_bonus(50) == 5


def test_combo_attack_in_game_flow():
    assert evaluate_clear(1, "none", 1, 0, False).attack == 0  # 1st clear
    assert evaluate_clear(1, "none", 2, 0, False).attack == 0  # 2nd clear
    assert evaluate_clear(1, "none", 3, 0, False).attack == 1  # 3rd: +1
    assert evaluate_clear(4, "none", 5, 0, False).attack == 5  # 4+1


def test_combo_score():
    assert evaluate_clear(1, "none", 1, 0, False).score == 100
    assert evaluate_clear(1, "none", 2, 0, False).score == 150  # +50
    assert evaluate_clear(1, "none", 3, 0, False).score == 200  # +100


def test_mini_double_counts_as_tsd():
    r = evaluate_clear(2, "mini", 1, 0, False)
    assert r.kind == "full"
    assert r.score == 1200
    assert r.attack == 4


def test_perfect_clear():
    r = evaluate_clear(4, "none", 1, 0, True)
    assert r.attack == 10
    assert r.score == 800 + 3000

    r2 = evaluate_clear(1, "none", 1, 0, True)
    assert r2.attack == 10  # replaces the (zero) base attack
    assert r2.score == 100 + 3000


def test_b2b_multiplier_does_not_apply_to_pc_bonus():
    r = evaluate_clear(4, "none", 1, 1, True)
    assert r.score == 1200 + 3000  # b2b quad 1200, PC added flat
    assert r.attack == 11  # PC 10 + b2b 1
