use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::{criterion_group, criterion_main};

fn drillx_compiled(nonces: u64) {
    let mut memory = drillx::equix::SolverMemory::new();
    let challenge = [255; 32];
    for nonce in 0..nonces {
        drillx::hash_with_shared_memory(&mut memory, &challenge, &nonce.to_le_bytes()).ok();
    }
}

fn different_sizes(c: &mut Criterion) {
    static KH: usize = 1000;
    let mut group = c.benchmark_group("from_elem");
    for size in [KH, 2 * KH, 4 * KH].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| drillx_compiled(size as u64))
        });
    }
    group.finish();
}

criterion_group!(benches, different_sizes);
criterion_main!(benches);
