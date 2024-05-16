#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=equix/");
    println!("cargo:rerun-if-changed=equix/hashx/");
    println!("cargo:rerun-if-changed=src/");

    // Compile hashx C code
    cc::Build::new()
        .include("equix/hashx/include")
        .include("equix/hashx/src")
        .include("equix/include")
        .include("equix/src")
        .file("equix/hashx/src/bench.c")
        .file("equix/hashx/src/context.c")
        .file("equix/hashx/src/solver.c")
        .file("equix/hashx/src/equix.c")
        .compile("hashx.a");

    // Compile equix C code
    cc::Build::new()
        .include("equix/include")
        .include("equix/src")
        .file("equix/src/bench.c")
        .file("equix/src/context.c")
        .file("equix/src/equix.c")
        .file("equix/src/solver.c")
        .compile("equix.a");

    cc::Build::new()
        .cuda(true)
        .include("equix/include")
        .include("equix/src")
        .include("equix/hashx/include")
        .include("equix/hashx/src")
        .file("cuda/drillx.cu")
        .flag("-cudart=static")
        // .flag("-gencode=arch=compute_89,code=sm_89") // Optimize for RTX 4090
        // .flag("-gencode=arch=compute_89,code=compute_89") // PTX for future compatibility
        .compile("drillx.a");

    // Add link directory
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Emit the location of the compiled library
    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir);
}
