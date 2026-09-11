//! Debug dumper: read a board state file (the exact bytes the trainer
//! feeds Blockfish) and print Blockfish's RAW analysis output — every
//! ranked root suggestion as its rating and its input sequence.
//!
//! Usage: cargo run --release --example dump -- <state-file>
//!
//! State file format (text):
//!   hold <C>        C = piece char or '.' for empty
//!   queue <chars>   pieces AFTER the active one is already included:
//!                   this is the exact array passed on the shim's wire
//!   rows
//!   <40 lines>      row 0 = top of our 40-row field, 'x' = filled
//!
//! The matrix is materialized from the occupied bottom band exactly like
//! bots/blockfish-shim/src/lib.rs does, so this is the same input.

use blockfish::ai::{AI, Snapshot};
use blockfish::{BasicMatrix, Color, Config};
use std::convert::TryFrom;

fn input_name(i: blockfish::Input) -> &'static str {
    match i {
        blockfish::Input::Left => "LEFT",
        blockfish::Input::Right => "RIGHT",
        blockfish::Input::CW => "CW",
        blockfish::Input::CCW => "CCW",
        blockfish::Input::Hold => "HOLD",
        blockfish::Input::SD => "SD",
        blockfish::Input::HD => "HD",
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump <state-file>");
    let text = std::fs::read_to_string(&path).expect("read state file");

    let mut hold: Option<Color> = None;
    let mut queue: Vec<Color> = vec![];
    let mut row_strs: Vec<String> = vec![];
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("hold ") {
            let ch = rest.trim().chars().next().unwrap_or('.');
            if ch != '.' {
                hold = Some(Color::try_from(ch).expect("bad hold char"));
            }
        } else if let Some(rest) = line.strip_prefix("queue ") {
            queue = rest
                .trim()
                .chars()
                .map(|c| Color::try_from(c).expect("bad queue char"))
                .collect();
        } else if line.starts_with("rows") {
            // header, rows follow
        } else if !line.trim().is_empty() {
            row_strs.push(line.trim().to_string());
        }
    }
    assert_eq!(row_strs.len(), 40, "expected 40 row lines");

    // same as the shim: only the occupied bottom band is materialized
    // (our row 0 = top, so the FIRST row containing 'x' is the band top)
    let top_occupied = row_strs
        .iter()
        .position(|s| s.contains('x'))
        .unwrap_or(40);
    let mut mat = BasicMatrix::with_cols(10);
    for k in 0..(40 - top_occupied) {
        let ours = &row_strs[39 - k];
        for (j, ch) in ours.chars().enumerate() {
            if ch == 'x' {
                mat.set((k as u16, j as u16));
            }
        }
    }

    println!("=== INPUT TO BLOCKFISH ===");
    println!("hold: {:?}", hold.map(|c| c.as_char()));
    println!("queue: {}", queue.iter().map(|c| c.as_char()).collect::<String>());
    println!("matrix height: {} rows (from our rows {}..39)", mat.rows(), top_occupied);
    for k in (0..mat.rows()).rev() {
        let s: String = (0..10).map(|j| if mat.get((k, j)) { 'x' } else { '.' }).collect();
        println!("  bf row {:>2}: {}", k, s);
    }
    println!();

    let mut ai = AI::new(Config::default());
    let mut handle = ai.analyze(Snapshot {
        hold,
        queue: queue.clone(),
        matrix: mat,
    });
    handle.wait();

    let mut ids = handle.all_moves().collect::<Vec<_>>();
    ids.sort_by(|&m, &n| handle.cmp(m, n));

    println!("=== RAW BLOCKFISH OUTPUT ===");
    println!("root moves: {}", ids.len());
    for (rank, &m) in ids.iter().enumerate() {
        let sugg = handle.suggestion(m, usize::MAX);
        let names: Vec<&str> = sugg.inputs.iter().map(|&i| input_name(i)).collect();
        let hds = sugg.inputs.iter().filter(|&&i| i == blockfish::Input::HD).count();
        println!("rank {:>2}: rating={} placements={} inputs=[{}]",
                 rank, sugg.rating, hds, names.join(" "));
    }
}
