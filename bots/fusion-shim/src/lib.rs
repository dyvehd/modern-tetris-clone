//! C ABI shim over fusion-engine (MochBot) for the cheese trainer.
//!
//! Exposes one entry point: given our engine's board state (rows, queue,
//! hold, active piece), run the heuristic beam search (no ONNX model —
//! `SearchRequest::runtime = None`, the same path bot_arena uses in
//! `--heuristic` mode) and report the best placement plus every scored
//! root candidate, as ABSOLUTE CELL SETS in our frame.
//!
//! Conversions (see fusion's src/header.rs, board.rs):
//!   - fusion board rows are y-up (row 0 = bottom), u16 bitmask, bit x =
//!     column x. Ours are y-down over 40 rows: our row i -> fusion row
//!     39 - i.
//!   - fusion `Piece` internal order I,O,T,L,J,S,Z; ours I,J,L,O,S,T,Z —
//!     mapped here, never bit-for-bit.
//!   - `Move::cells()` are OFFSETS from the pivot; the returned candidate
//!     struct carries the four absolute cells already computed.
//!
//! Like every trainer backend, fusion only ever sees placements; our own
//! pathfinder navigates them.

use fusion_engine::board::Board;
use fusion_engine::eval::EvalWeights;
use fusion_engine::header::{piece_from_external, Piece};
use fusion_engine::search::{self, SearchRequest};
use fusion_engine::search_config::SearchConfig;
use fusion_engine::state::GameState;

// our PieceType value -> fusion internal Piece
fn piece_from_ours(v: u8) -> Option<Piece> {
    match v {
        // ours: I=0 J=1 L=2 O=3 S=4 T=5 Z=6
        0 => Some(Piece::I),
        1 => Some(Piece::J),
        2 => Some(Piece::L),
        3 => Some(Piece::O),
        4 => Some(Piece::S),
        5 => Some(Piece::T),
        6 => Some(Piece::Z),
        _ => None,
    }
}

fn piece_to_ours(p: Piece) -> u8 {
    match p {
        Piece::I => 0,
        Piece::J => 1,
        Piece::L => 2,
        Piece::O => 3,
        Piece::S => 4,
        Piece::T => 5,
        Piece::Z => 6,
    }
}

#[repr(C)]
pub struct FCandidate {
    pub hold: i32,
    pub piece_id: i32,   // our PieceType value
    pub cells_r: [i32; 4],
    pub cells_c: [i32; 4],
    pub score: f32,      // fusion's root score
}

#[repr(C)]
pub struct FResult {
    pub ok: i32,
    pub hold: i32,
    pub piece_id: i32,
    pub cells_r: [i32; 4],
    pub cells_c: [i32; 4],
    pub n_cands: i32,          // candidates written
    pub complexity: f32,       // search's position_complexity
    pub best_score: f32,
}

#[no_mangle]
pub extern "C" fn fs_version() -> i32 {
    1
}

fn build_board(rows: &[u16; 40]) -> Board {
    let mut board = Board::new();
    for (i, &row) in rows.iter().enumerate() {
        let y = 39 - i; // our row 0 (top) -> fusion row 39
        board.rows[y] = row;
    }
    // rebuild column bitboards (board_from_u16's loop, coach_beam.rs:168)
    for x in 0..10 {
        board.cols[x] = 0;
    }
    for (y, &row) in board.rows.iter().enumerate() {
        for x in 0..10 {
            if row >> x & 1 == 1 {
                board.cols[x] |= 1u64 << y;
            }
        }
    }
    board
}

fn cells_of(mv: fusion_engine::header::Move) -> [i32; 8] {
    // pivot + 3 offsets, in fusion y-up coords; converted to our (row, col)
    let mut out = [0i32; 8];
    let px = mv.x() as i32;
    let py = mv.y() as i32;
    out[0] = px;
    out[1] = py;
    for (i, c) in mv.cells().coords.iter().enumerate() {
        out[2 + 2 * i] = px + c.x as i32;
        out[3 + 2 * i] = py + c.y as i32;
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn fs_think(
    rows: *const u16,           // 40 row bitmasks, ours (row 0 = top)
    n_queue: i32,
    queue_ext: *const u8,       // external order I,O,T,S,Z,J,L per piece_from_external
    hold_ext: i32,              // -1 = none, else external id
    current_ours: u8,           // our PieceType value
    b2b: u32,
    combo: u32,
    out_best: *mut FResult,
    cands: *mut FCandidate,
    cap_cands: i32,
    beam_width: i32,
    depth: i32,
) -> i32 {
    let rows_slice = std::slice::from_raw_parts(rows, 40);
    let rows_arr: [u16; 40] = {
        let mut a = [0u16; 40];
        a.copy_from_slice(rows_slice);
        a
    };

    let current = match piece_from_ours(current_ours) {
        Some(p) => p,
        None => return 0,
    };
    let queue: Vec<Piece> = std::slice::from_raw_parts(queue_ext, n_queue as usize)
        .iter()
        .filter_map(|&v| piece_from_external(v))
        .collect();
    let hold = if hold_ext < 0 {
        None
    } else {
        piece_from_external(hold_ext as u8)
    };

    let mut state = GameState::new(build_board(&rows_arr), current, queue);
    state.hold = hold;
    state.b2b = b2b as u8;
    state.combo = combo;

    let config = SearchConfig {
        beam_width: if beam_width > 0 { beam_width as usize } else { 160 },
        depth: if depth > 0 { depth as usize } else { 6 },
        extend_queue_7bag: true,
        ..Default::default()
    };
    let weights = EvalWeights::default();
    let request = SearchRequest {
        config: &config,
        weights: &weights,
        runtime: None, // heuristic mode: no ONNX model
        forced_root_move: None,
    };

    let result = match search::search(&state, &request) {
        Some(r) => r,
        None => return 0,
    };

    let best = &result.best;
    let cells = cells_of(best.best_move);
    // convert y-up (col, y) cells to our (row, col)
    let mut ours_r = [0i32; 4];
    let mut ours_c = [0i32; 4];
    for i in 0..4 {
        let cx = cells[2 * i];
        let cy = cells[2 * i + 1];
        ours_c[i] = cx;
        ours_r[i] = 39 - cy;
    }

    let mut n_out = 0i32;
    if !cands.is_null() && cap_cands > 0 {
        let buf = std::slice::from_raw_parts_mut(cands, cap_cands as usize);
        for (mv, score) in result.root_scores.iter() {
            if n_out as usize >= buf.len() {
                break;
            }
            let cs = cells_of(*mv);
            let mut cr = [0i32; 4];
            let mut cc = [0i32; 4];
            let mut ok = true;
            for i in 0..4 {
                cc[i] = cs[2 * i];
                cr[i] = 39 - cs[2 * i + 1];
                if cr[i] < 0 || cr[i] > 39 || cc[i] < 0 || cc[i] > 9 {
                    ok = false;
                }
            }
            if !ok {
                continue;
            }
            buf[n_out as usize] = FCandidate {
                hold: best.hold_used as i32,
                piece_id: piece_to_ours(mv.piece()) as i32,
                cells_r: cr,
                cells_c: cc,
                score: *score,
            };
            n_out += 1;
        }
    }

    if !out_best.is_null() {
        *out_best = FResult {
            ok: 1,
            hold: best.hold_used as i32,
            piece_id: piece_to_ours(best.best_move.piece()) as i32,
            cells_r: ours_r,
            cells_c: ours_c,
            n_cands: n_out,
            complexity: result.position_complexity,
            best_score: best.score,
        };
    }
    1
}
