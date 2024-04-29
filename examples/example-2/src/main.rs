use std::{collections::HashMap, time::Instant};

fn main() {
    // Current challenge (255s for demo)
    let challenge = [255; 32];

    // Do work
    let timer = Instant::now();
    let mut hash_count = HashMap::<u32, u64>::new();
    let mut nonce: u64 = 0;
    loop {
        // Track difficulties
        let hx = drillx::hash(&challenge, &nonce.to_le_bytes());
        let d = drillx::difficulty(hx);
        hash_count.insert(d, hash_count.get(&d).unwrap_or(&0).saturating_add(1));
        if nonce % 1000 == 0 {
            print(&hash_count, &timer);
        }

        // Increment nonce
        nonce += 1;
    }
}

fn print(hash_counts: &HashMap<u32, u64>, timer: &Instant) {
    let mut str = format!("{} sec – ", timer.elapsed().as_secs());
    for i in 0..hash_counts.len() as u32 {
        str = format!("{}{}: {} ", str, i, hash_counts.get(&i).unwrap_or(&0)).to_string();
    }
    println!("{}", str);
}
