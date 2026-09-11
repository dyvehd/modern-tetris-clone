//! C ABI shim over blockfish-engine (iitalics) for the cheese trainer.
//!
//! One entry point: given our engine's board state, run a Blockfish
//! analysis synchronously and return EVERY ranked root candidate
//! (rating — lower is better — plus absolute lock cells) plus the best
//! candidate's multi-placement plan (the first `plan_len` placements in
//! its trace), all as ABSOLUTE CELL SETS in our frame.
//!
//! Conversions:
//!   - blockfish's BasicMatrix is y-up (row 0 = bottom, bit j = column j,
//!     arbitrary height); our rows are y-down over 40 rows: our row i ->
//!     blockfish row 39 - i. Only the occupied bottom band is
//!     materialized.
//!   - Pieces cross the boundary as BLOCKFISH COLOR CHARS
//!     ('I','J','L','O','S','T','Z'); ours PieceType value v -> "IJLOSTZ"[v].
//!   - The shape table is rebuilt from the engine's own SRS data file by
//!     deserializing into `blockfish::ShapeTable` — the same bytes
//!     `shape::srs()` uses, reachable from outside the crate because
//!     ShapeTable is re-exported under the "gen-shtb" feature.
//!
//! Recovering placements: the public engine surface reports suggestions
//! as INPUT sequences ([Hold?] + finesse moves + HD per placement), so
//! — like upstream `reconstruct_inputs` — we replay them: spawn is
//! `(matrix.rows(), shape.spawn_col(), R0)`, movements/rotations go
//! through `ShapeRef::try_input` (SRS kicks), SD/HD land via
//! `sonic_drop`, the lock cells are the stage diff BEFORE `sift_rows`
//! (line clears) runs, and the engine's own queue/hold bookkeeping
//! (state.rs `next`/`pop`) is mirrored between placements.
//!
//! Like every trainer backend, blockfish only ever outputs placements;
//! our own pathfinder navigates them.

use blockfish::ai::{AI, Snapshot};
use blockfish::{BasicMatrix, Color, Orientation, ShapeTable};

use std::convert::TryFrom;

const PLAN_STEPS: usize = 8;

#[repr(C)]
pub struct BFCandidate {
    pub hold: i32,
    pub piece: u8,            // color char, e.g. b'T' (0 = failed)
    pub cells_r: [i32; 4],    // ours (row, col), row 0 = top
    pub cells_c: [i32; 4],
    pub rating: i64,          // blockfish rating (lower = better)
    pub n_plan: i32,          // plan steps after this candidate's FIRST move
    pub plan_piece: [u8; 8],
    pub plan_hold: [i32; 8],
    pub plan_r: [[i32; 4]; 8],
    pub plan_c: [[i32; 4]; 8],
    pub n_inputs: i32,        // raw suggestion input count
    pub inputs: [u8; 256],    // raw input codes (Input enum order per proto)
}

impl Default for BFCandidate {
    fn default() -> Self {
        BFCandidate {
            hold: 0,
            piece: 0,
            cells_r: [0; 4],
            cells_c: [0; 4],
            rating: 0,
            n_plan: 0,
            plan_piece: [0; 8],
            plan_hold: [0; 8],
            plan_r: [[0; 4]; 8],
            plan_c: [[0; 4]; 8],
            n_inputs: 0,
            inputs: [0; 256],
        }
    }
}

#[repr(C)]
#[derive(Default)]
pub struct BFResult {
    pub ok: i32,
    pub n_cand: i32,
    pub nodes: u64,
    pub iterations: u64,
    pub millis: u64,
}

#[no_mangle]
pub extern "C" fn bf_version() -> i32 {
    2
}

/// Our 40-row y-down board -> a blockfish BasicMatrix (y-up, only the
/// occupied bottom band).
fn matrix_from(rows: &[u16]) -> BasicMatrix {
    // our row 0 is the TOP, so the topmost occupied row is the FIRST
    // non-empty index; the band we materialize spans [top_occupied, 39].
    let top_occupied = rows.iter().position(|&r| r != 0).unwrap_or(40);
    let mut mat = BasicMatrix::with_cols(10);
    for k in 0..(40 - top_occupied) {
        let ours_row = rows[39 - k];
        for j in 0..10 {
            if ours_row >> j & 1 == 1 {
                mat.set((k as u16, j as u16));
            }
        }
    }
    mat
}

fn clone_occupied(mat: &BasicMatrix) -> BasicMatrix {
    let mut out = BasicMatrix::with_cols(10);
    for i in 0..mat.rows() {
        for j in 0..10 {
            if mat.get((i, j)) {
                out.set((i, j));
            }
        }
    }
    out
}

/// The queue/hold bookkeeping of the analysis state, mirroring
/// blockfish-engine ai/state.rs VERBATIM: the queue is stored reversed
/// (next piece at the end), and the hold piece — if the slot is occupied —
/// sits ON TOP of it (the last element). With the slot empty, `next()`
/// reports the 2nd queue piece as the piece a hold would bring out.
struct QueueState {
    queue_rev: Vec<Color>,
    has_held: bool,
}

impl QueueState {
    fn new(queue: &[Color], hold: Option<Color>) -> Self {
        // ai/state.rs `From<Snapshot> for State`
        let mut queue_rev = queue.iter().rev().cloned().collect::<Vec<_>>();
        let mut has_held = false;
        if let Some(h) = hold {
            has_held = true;
            queue_rev.push(h);
        }
        QueueState { queue_rev, has_held }
    }

    fn next(&self) -> (Option<Color>, Option<Color>) {
        // ai/state.rs `State::next`: `(next_piece, hold_piece)` where
        // hold_piece is what a hold BRINGS OUT (the 2nd queue piece when
        // the slot is empty), not necessarily the piece sitting in hold.
        let from_top = |i: usize| {
            self.queue_rev
                .len()
                .checked_sub(i)
                .and_then(|i| self.queue_rev.get(i))
                .cloned()
        };
        let c1 = from_top(1);
        let c2 = from_top(2);
        if self.has_held {
            (c2, c1)
        } else {
            (c1, c2)
        }
    }

    fn pop(&mut self, hold: bool) {
        // ai/state.rs `State::pop`:
        //   | has_held | hold | pos |
        //   | true     | true |  1  |
        //   | true     | false|  2  |
        //   | false    | true |  2  |
        //   | false    | false|  1  |
        let pos = if self.has_held == hold { 1 } else { 2 };
        if let Some(idx) = self.queue_rev.len().checked_sub(pos) {
            self.queue_rev.remove(idx);
        }
        self.has_held |= hold;
    }
}

/// One placement recovered by replaying a suggestion segment.
struct PlacementOut {
    piece: Color,
    did_hold: bool,
    /// absolute lock cells in OUR frame (row 0 = top)
    cells: Vec<(i32, i32)>,
}

/// Replay a suggestion's input sequence ([Hold?] + finesse moves + HD per
/// placement) from the snapshot state, recovering each placement's lock
/// cells, up to `max_steps` placements. Mirrors `reconstruct_inputs` +
/// `State::place` from blockfish-engine. A Hold is only honored while OUR
/// engine could perform it — hold_enabled off, locked for the active
/// piece (`hold_first`, after the player held), or locked for the piece a
/// hold just brought out (the placement right after one that held) —
/// otherwise the suggestion is unplayable here (Err).
fn replay_suggestion(
    shtb: &ShapeTable,
    mat: &BasicMatrix,
    queue: &[Color],
    hold: Option<Color>,
    sugg: &[blockfish::Input],
    max_steps: usize,
    hold_first: bool,
    hold_feature: bool,
) -> Result<Vec<PlacementOut>, ()> {
    let mut state = QueueState::new(queue, hold);
    let mut mat_curr = clone_occupied(mat);
    let mut out: Vec<PlacementOut> = vec![];
    let mut seg: Vec<blockfish::Input> = vec![];
    for &input in sugg.iter() {
        if out.len() >= max_steps {
            break;
        }
        if input != blockfish::Input::HD {
            seg.push(input);
            continue;
        }
        // one placement: [Hold?] + moves + HD
        //
        // A hold is legal for us only while the engine would allow it: at
        // the FIRST placement that means can_hold (hold not already used
        // for this piece); at every later placement the preceding lock has
        // re-armed hold, so it is always allowed.
        let step = out.len();
        let hold_ok = hold_feature && (step > 0 || hold_first);
        let (next_nh, next_h) = state.next();
        let did_hold = matches!(seg.first(), Some(&blockfish::Input::Hold));
        if did_hold && !hold_ok {
            return Err(());
        }
        let color = match if did_hold { next_h } else { next_nh } {
            Some(c) => c,
            None => return Err(()),
        };
        let shape = match shtb.shape(color) {
            Some(s) => s,
            None => return Err(()),
        };
        let moves: &[blockfish::Input] = if did_hold { &seg[1..] } else { &seg[..] };
        // simulate from spawn (FinesseFinder's starting transform)
        let mut tf = (mat_curr.rows() as i16, shape.spawn_col(), Orientation::R0);
        for &mv in moves {
            tf = match mv {
                blockfish::Input::SD => shape.sonic_drop(&mat_curr, tf),
                blockfish::Input::Left
                | blockfish::Input::Right
                | blockfish::Input::CW
                | blockfish::Input::CCW => match shape.try_input(&mat_curr, tf, mv) {
                    Some(t) => t,
                    None => return Err(()),
                },
                _ => return Err(()),
            };
        }
        let final_tf = shape.sonic_drop(&mat_curr, tf);
        // lock cells = the stage diff BEFORE clears (sift_rows)
        let mut after = clone_occupied(&mat_curr);
        shape.blit_to(&mut after, final_tf);
        let mut cells: Vec<(i32, i32)> = vec![];
        for i in 0..after.rows() {
            for j in 0..10 {
                if !mat_curr.get((i, j)) && after.get((i, j)) {
                    cells.push((39 - i as i32, j as i32));
                }
            }
        }
        if cells.len() != 4 {
            return Err(());
        }
        out.push(PlacementOut { piece: color, did_hold, cells });
        seg.clear();
        // apply the placement (State::place): blit, clear lines, pop queue
        let mut placed = clone_occupied(&mat_curr);
        shape.blit_to(&mut placed, final_tf);
        placed.sift_rows();
        mat_curr = placed;
        state.pop(did_hold);
    }
    if out.is_empty() {
        return Err(());
    }
    Ok(out)
}

/// `search_limit` is blockfish's node budget (its Config.search_limit; its
/// CLI counts kilo-nodes), `plan_len` caps the best move's plan depth
/// (>= 1: the chosen move plus plan_len-1 future placements).
///
/// `hold_ch` is the TRUE piece in the hold slot (0 = empty) even while
/// hold is locked for the active piece — blockfish needs it to model
/// future swaps; candidates that hold while locked are dropped here.
///
/// Returns 1 on success (and fills `*out`, writing up to `cap_cands`
/// candidates ranked best-first into `cands`), 0 on failure. A candidate
/// that fails to replay (unplayable hold, movement mismatch, a blockfish
/// panic in input reconstruction) is skipped — it never poisons the rest
/// of the ranked list.
#[no_mangle]
pub unsafe extern "C" fn bf_think(
    rows: *const u16,      // 40 y-down row bitmasks
    hold_ch: u8,           // color char of the piece in the hold SLOT, 0 = empty
    hold_enabled: i32,     // 0 = the engine has no hold feature
    can_hold: i32,         // 0 = hold locked for the active piece
    n_queue: i32,
    queue_ch: *const u8,   // color chars, [active, preview...] (Snapshot queue)
    search_limit: i32,
    plan_len: i32,
    out: *mut BFResult,
    cands: *mut BFCandidate,
    cap_cands: i32,
) -> i32 {
    // a panic on the analysis path must never unwind into Python's
    // process (blockfish's input reconstruction uses `.expect`)
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bf_think_impl(
            rows, hold_ch, hold_enabled, can_hold, n_queue, queue_ch,
            search_limit, plan_len, out, cands, cap_cands,
        )
    }));
    r.unwrap_or(0)
}

unsafe fn bf_think_impl(
    rows: *const u16,
    hold_ch: u8,
    hold_enabled: i32,
    can_hold: i32,
    n_queue: i32,
    queue_ch: *const u8,
    search_limit: i32,
    plan_len: i32,
    out: *mut BFResult,
    cands: *mut BFCandidate,
    cap_cands: i32,
) -> i32 {
    let rows_slice = unsafe { std::slice::from_raw_parts(rows, 40) };
    let queue_slice: &[u8] = if n_queue > 0 {
        unsafe { std::slice::from_raw_parts(queue_ch, n_queue as usize) }
    } else {
        &[]
    };

    let shtb = std::sync::Arc::new(match serde_json::from_slice::<ShapeTable>(
        include_bytes!("../../../tmp/blockfish/support/test/srs-shape-table.json"),
    ) {
        Ok(t) => t,
        Err(_) => return 0,
    });

    let snap_hold = if hold_enabled == 0 || hold_ch == 0 {
        None
    } else {
        match Color::try_from(hold_ch as char) {
            Ok(c) => Some(c),
            Err(_) => return 0,
        }
    };
    let queue: Vec<Color> = queue_slice
        .iter()
        .filter_map(|&ch| Color::try_from(ch as char).ok())
        .collect();

    let mat = matrix_from(rows_slice);

    let cfg = blockfish::Config {
        search_limit: if search_limit > 0 {
            search_limit as usize
        } else {
            50_000
        },
        ..blockfish::Config::default()
    };
    let mut ai = AI::new(cfg);
    let mut handle = ai.analyze(Snapshot {
        hold: snap_hold,
        queue: queue.clone(),
        matrix: mat.clone(),
    });
    handle.wait();

    // rank all root moves best-first (blockfish ratings: lower = better)
    let mut ids = handle.all_moves().collect::<Vec<_>>();
    ids.sort_by(|&m, &n| handle.cmp(m, n));

    let mut n_out: i32 = 0;
    if !cands.is_null() && cap_cands > 0 && !ids.is_empty() {
        let buf = unsafe { std::slice::from_raw_parts_mut(cands, cap_cands as usize) };
        for &m_id in ids.iter() {
            if n_out as usize >= buf.len() {
                break;
            }
            let is_rank0 = n_out == 0;
            let max_steps = if is_rank0 {
                (plan_len.max(1) as usize).min(PLAN_STEPS + 1)
            } else {
                1
            };
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let sugg = handle.suggestion(m_id, max_steps);
                let full = handle.suggestion(m_id, usize::MAX);
                let res = replay_suggestion(
                    &shtb,
                    &mat,
                    &queue,
                    snap_hold,
                    &sugg.inputs,
                    max_steps,
                    can_hold != 0,
                    hold_enabled != 0,
                );
                (res, sugg.rating, full.inputs)
            }));
            let (res, rating, full_inputs) = match attempt {
                Ok(v) => v,
                Err(_) => continue, // panicking suggestion: skip, don't abort
            };
            let placements = match res {
                Ok(p) => p,
                Err(_) => continue, // unplayable here: hold locked, bad moves
            };
            let slot: *mut BFCandidate = &mut buf[n_out as usize];
            unsafe {
                *slot = BFCandidate::default();
                (*slot).rating = rating;
                (*slot).n_inputs = full_inputs.len().min(255) as i32;
                for (k, &inp) in full_inputs.iter().take(255).enumerate() {
                    (*slot).inputs[k] = inp as u8; // Input enum discriminants
                }
                let first = &placements[0];
                (*slot).hold = first.did_hold as i32;
                (*slot).piece = first.piece.as_char() as u8;
                for k in 0..4 {
                    (*slot).cells_r[k] = first.cells.get(k).map(|c| c.0).unwrap_or(-1);
                    (*slot).cells_c[k] = first.cells.get(k).map(|c| c.1).unwrap_or(-1);
                }
                // deeper placements of the best move's traced plan
                (*slot).n_plan = (placements.len() - 1).min(PLAN_STEPS) as i32;
                for (k, pl) in placements.iter().skip(1).take(PLAN_STEPS).enumerate() {
                    (*slot).plan_piece[k] = pl.piece.as_char() as u8;
                    (*slot).plan_hold[k] = pl.did_hold as i32;
                    for c in 0..4 {
                        (*slot).plan_r[k][c] = pl.cells.get(c).map(|c| c.0).unwrap_or(-1);
                        (*slot).plan_c[k][c] = pl.cells.get(c).map(|c| c.1).unwrap_or(-1);
                    }
                }
            }
            n_out += 1;
        }
    }

    if !out.is_null() {
        let stats = handle.stats().unwrap_or_default();
        *out = BFResult {
            ok: 1,
            n_cand: n_out,
            nodes: stats.nodes as u64,
            iterations: stats.iterations as u64,
            millis: stats.time_taken.as_millis() as u64,
        };
    }
    1
}
