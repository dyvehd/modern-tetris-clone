use blockfish::ai::{AI, Snapshot};
use blockfish::{BasicMatrix, Color, Config};
use std::convert::TryFrom;

fn main() {
    // a 9-row cheese stack with a hole column, like the trainer board
    let mut rows = vec![0u16; 40];
    let cheese: [u16; 9] = [
        0b1111111011, 0b1111110111, 0b1111101111, 0b1111011111, 0b1110111111, 0b1101111111,
        0b1011111111, 0b0111111111, 0b1111111101,
    ];
    for (k, bits) in cheese.iter().rev().enumerate() {
        rows[39 - k] = *bits;
    }
    let mut mat = BasicMatrix::with_cols(10);
    for (k, &row) in rows.iter().enumerate() {
        for j in 0..10 {
            if row >> j & 1 == 1 {
                mat.set((k as u16, j as u16));
            }
        }
    }
    let queue: Vec<Color> = "JSILT".chars().map(|c| Color::try_from(c).unwrap()).collect();
    let cfg = Config::default();
    let mut ai = AI::new(cfg);
    let mut handle = ai.analyze(Snapshot {
        hold: None,
        queue,
        matrix: mat.clone(),
    });
    handle.wait();
    let mut ids = handle.all_moves().collect::<Vec<_>>();
    ids.sort_by(|&m, &n| handle.cmp(m, n));
    println!("moves: {}", ids.len());
    for &m in ids.iter().take(5) {
        let s = handle.suggestion(m, 0);
        let full = handle.suggestion(m, usize::MAX);
        println!(
            "rating {} inputs {} (hd {})",
            s.rating,
            full.inputs.len(),
            full.inputs.iter().filter(|&&i| i == blockfish::Input::HD).count()
        );
    }
}
