"""Hold rules + the TGM-style IHS/IRS initial inputs (off by default).

IRS/IHS exist only in the TGM series; Jstris, TETR.IO, Nullpomino and
Techmino do not rotate/hold a piece because the key was still held when
the piece spawned (user-verified 2026-09-07 against real games).
"""

from conftest import force_queue, hard_drop, make_game, place

from tetris.engine.constants import PieceType
from tetris.engine.game import Action, Btn

# T, then the rest — several IRS tests place obstacles for a T piece
_QUEUE = [PieceType.T] + [p for p in PieceType if p is not PieceType.T]


def test_hold_swap_and_once_per_piece():
    game = make_game()
    game.spawn_forced(PieceType.T)
    first_next = game.queue[0]
    game.tick([Action.HOLD])
    assert game.hold_type is PieceType.T
    assert game.active.type is first_next
    assert game.can_hold is False
    game.tick([Action.HOLD])  # ignored: hold already used
    assert game.hold_type is PieceType.T
    assert game.active.type is first_next

    # locking restores hold availability
    hard_drop(game)
    assert game.can_hold is True


def test_hold_swapped_piece_spawns_fresh():
    game = make_game()
    place(game, PieceType.T, x=4, y=30, rot=2)  # mid-air, rotated
    game.tick([Action.HOLD])
    p = game.active
    assert (p.rot, p.x, p.y) == (0, 3, 18)  # fresh spawn state


def test_hold_empty_slot_pulls_from_queue():
    game = make_game()
    game.spawn_forced(PieceType.S)
    expected_next = game.queue[0]
    game.tick([Action.HOLD])
    assert game.hold_type is PieceType.S
    assert game.active.type is expected_next


# --- held keys must NOT carry over to the next piece (default) ----------------


def test_rotate_key_held_across_hard_drop_does_not_irs():
    """Replays the user's Sprint log (2026-09-07): rotate CW pressed once,
    key released ~1.5 s later — every piece spawned in between came out
    with 1 CW rotation. Spawn rotation from a held key is TGM-only."""
    game = make_game()
    force_queue(game, _QUEUE)
    game.tick([Action.ROT_CW], frozenset({Btn.ROT_CW}))  # rotate the first piece
    assert game.active.rot == 1
    hard_drop(game, frozenset({Btn.ROT_CW}))  # drop while still holding CW
    p = game.active
    assert p is not None and p.rot == 0  # next piece spawns unrotated
    hard_drop(game, frozenset({Btn.ROT_CW}))  # and the one after that
    assert game.active.rot == 0


def test_hold_key_held_across_hard_drop_does_not_ihs():
    """The same carryover for hold: keeping the hold key down must not
    hold every spawned piece (IHS is TGM-only)."""
    game = make_game()
    game.spawn_forced(PieceType.J)
    hard_drop(game)
    incoming = game.queue[0]
    game.tick([Action.HARD_DROP], frozenset({Btn.HOLD}))
    assert game.active.type is incoming  # spawned normally, not swapped
    assert game.hold_type is None
    assert game.can_hold is True


# --- IRS/IHS as opt-in TGM-style flags ---------------------------------------


def test_irs_opt_in_rotates_at_spawn():
    game = make_game(irs_enabled=True)
    force_queue(game, _QUEUE)
    game.tick([], frozenset({Btn.ROT_CW}))  # first spawn with CW held
    assert game.active.rot == 1
    assert game.active.y == 18


def test_irs_opt_in_ccw_and_180():
    game = make_game(irs_enabled=True)
    force_queue(game, _QUEUE)
    game.tick([], frozenset({Btn.ROT_CCW}))
    assert game.active.rot == 3
    # ARE is 0, so the next piece spawns during this same hard-drop tick:
    # the 180 button must already be held then.
    hard_drop(game, frozenset({Btn.ROT_180}))
    assert game.active.rot == 2


def test_irs_opt_in_uses_wall_kicks():
    # (4,19) occupied blocks the basic IRS rotation; kick (-1,+1) applies
    # and the piece still spawns rotated.
    game = make_game(irs_enabled=True)
    force_queue(game, _QUEUE)
    game.rows[19] |= 1 << 4
    game.tick([], frozenset({Btn.ROT_CW}))
    assert game.active.rot == 1
    assert (game.active.x, game.active.y) == (2, 17)


def test_irs_opt_in_cannot_save_a_sealed_spawn():
    game = make_game(irs_enabled=True)
    force_queue(game, _QUEUE)
    # Seal off the spawn area plus all downward kick space so even the
    # IRS cannot save the piece.
    for row in range(18, 23):
        game.rows[row] = 0b1111111111
    game.tick([], frozenset({Btn.ROT_CW}))
    assert game.over  # no room even after the IRS kicks


def test_ihs_opt_in_initial_hold_at_spawn():
    game = make_game(ihs_enabled=True)
    game.spawn_forced(PieceType.J)
    hard_drop(game)
    # active piece = queue[0]; after it locks, the incoming piece (queue[0])
    # spawns and IHS swaps it into hold, releasing queue[1].
    incoming, after = game.queue[0], game.queue[1]
    game.tick([Action.HARD_DROP], frozenset({Btn.HOLD}))
    assert game.hold_type is incoming
    assert game.active.type is after
    assert game.can_hold is False
