#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=src/");

    // Compile hashx C code
    cc::Build::new()
        .cuda(true)
        .compiler("nvcc")
        .no_default_flags(true)
        .include("cuda/equix/hashx/include")
        .include("cuda/equix/hashx/src")
        .file("cuda/equix/hashx/src/blake2.c")
        .file("cuda/equix/hashx/src/compiler.c")
        .file("cuda/equix/hashx/src/compiler_a64.c")
        .file("cuda/equix/hashx/src/compiler_x86.c")
        .file("cuda/equix/hashx/src/context.c")
        .file("cuda/equix/hashx/src/hashx.c")
        .file("cuda/equix/hashx/src/hashx_thread.c")
        .file("cuda/equix/hashx/src/program.c")
        .file("cuda/equix/hashx/src/program_exec.c")
        .file("cuda/equix/hashx/src/siphash.c")
        .file("cuda/equix/hashx/src/siphash_rng.c")
        .file("cuda/equix/hashx/src/virtual_memory.c")
        .flag("-O0")
        .flag("-fdata-sections")
        .flag("-fPIC")
        .flag("-G")
        .flag("-gdwarf-4")
        .flag("-fno-omit-frame-pointer")
        .flag("-m64")
        .compile("hashx.a");

    // Compile equix C code
    cc::Build::new()
        .cuda(true)
        .compiler("nvcc")
        .no_default_flags(true)
        .include("cuda/equix/include")
        .include("cuda/equix/src")
        .include("cuda/equix/hashx/include")
        .include("cuda/equix/hashx/src")
        .file("cuda/equix/src/context.c")
        .file("cuda/equix/src/equix.c")
        .file("cuda/equix/src/solver.c")
        .flag("-O0")
        .flag("-fdata-sections")
        .flag("-fPIC")
        .flag("-G")
        .flag("-gdwarf-4")
        .flag("-fno-omit-frame-pointer")
        .flag("-m64")
        .compile("equix.a");

    // Compile drillx
    cc::Build::new()
        .cuda(true)
        .include("cuda/equix/include")
        .include("cuda/equix/hashx/include")
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
