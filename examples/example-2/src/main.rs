use std::{collections::HashMap, time::Instant};

use drillx::equix::SolverMemory;

fn main() {
    let timer = Instant::now();
    let challenge = [255; 32];
    let mut memory = SolverMemory::new();
    let mut hash_count = HashMap::<u32, u64>::new();
    let mut nonce: u64 = 0;
    loop {
        // Track difficulties
        if let Ok(hx) = drillx::hash_with_memory(&mut memory, &challenge, &nonce.to_le_bytes()) {
            let diff = hx.difficulty();
            hash_count.insert(diff, hash_count.get(&diff).unwrap_or(&0).saturating_add(1));
            if nonce % 100 == 0 {
                print(&hash_count, &timer);
            }
        }

        // Increment nonce
        nonce += 1;
    }
}

fn print(hash_counts: &HashMap<u32, u64>, timer: &Instant) {
    let max_key = *hash_counts.keys().max().unwrap();
    let mut str = format!("{} sec – ", timer.elapsed().as_secs());
    for i in 0..(max_key + 1) {
        str = format!("{} {}: {} ", str, i, hash_counts.get(&i).unwrap_or(&0)).to_string();
    }
    println!("{}", str);
}
