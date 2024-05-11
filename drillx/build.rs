#[cfg(not(feature = "gpu"))]
fn main() {}

#[cfg(feature = "gpu")]
fn main() {
    gpu_keccak();
}

#[cfg(feature = "gpu")]
fn gpu_keccak() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=src/");

    cc::Build::new()
        .cuda(true)
        .file("kernels/keccak.cu")
        .file("kernels/utils.cu")
        .file("kernels/orev2.cu")
        .flag("-cudart=static")
        // .flag("-gencode=arch=compute_89,code=sm_89") // Optimize for RTX 4090
        // .flag("-gencode=arch=compute_89,code=compute_89") // PTX for future compatibility
        .compile("libkeccak.a");

    // Add link directory
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Emit the location of the compiled library
    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir);
}
