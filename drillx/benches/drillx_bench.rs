use criterion::{
    BenchmarkId, Criterion, Throughput, {criterion_group, criterion_main},
};

fn drillx_loop(nonces: u64) {
    let mut memory = drillx::equix::SolverMemory::new();
    let challenge = [255; 32];
    for nonce in 0..nonces {
        drillx::hash_with_memory(&mut memory, &challenge, &nonce.to_le_bytes()).ok();
    }
}

fn different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("drillx");
    for size in [1, 10, 100].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| drillx_loop(size as u64))
        });
    }
    group.finish();
}

criterion_group!(benches, different_sizes);
criterion_main!(benches);
