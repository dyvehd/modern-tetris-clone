"""7-bag randomizer properties."""

from tetris.engine.constants import ALL_PIECES, PieceType
from tetris.engine.rng import SevenBag


def test_every_window_of_seven_is_a_full_bag():
    bag = SevenBag(seed=42)
    pieces = [bag.next_piece() for _ in range(700)]
    # The 7-bag guarantee: each disjoint chunk of 7 deals contains every
    # piece exactly once. (A *sliding* window may span a bag boundary and
    # legitimately repeat the previous bag's last piece.)
    for i in range(0, len(pieces), 7):
        assert set(pieces[i : i + 7]) == set(ALL_PIECES)


def test_seed_determinism():
    a = [SevenBag(seed=7).next_piece() for _ in range(100)]
    b = [SevenBag(seed=7).next_piece() for _ in range(100)]
    assert a == b


def test_different_seeds_diverge():
    a = [SevenBag(seed=1).next_piece() for _ in range(50)]
    b = [SevenBag(seed=2).next_piece() for _ in range(50)]
    assert a != b


def test_peek_does_not_consume():
    bag = SevenBag(seed=3)
    preview = bag.peek(5)
    assert len(preview) == 5
    dealt = [bag.next_piece() for _ in range(5)]
    assert dealt == preview
