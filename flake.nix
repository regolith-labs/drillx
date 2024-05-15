{
  description = "Drillix development environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # Rust
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };
        lib = pkgs.lib;
        toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      in
      {
        devShells.default = pkgs.mkShell
          {
            name = "ore-cli";
            nativeBuildInputs = [
              pkgs.pkg-config
              pkgs.clang
              # pkgs.zluda
              # pkgs.cudaPackages.cuda_nvcc
              # Mold Linker for faster builds (only on Linux)
              (lib.optionals pkgs.stdenv.isLinux pkgs.mold)
              (lib.optionals pkgs.stdenv.isDarwin pkgs.darwin.apple_sdk.frameworks.Security)
              (lib.optionals pkgs.stdenv.isDarwin pkgs.darwin.apple_sdk.frameworks.SystemConfiguration)
            ];
            buildInputs = [
              # We want the unwrapped version, wrapped comes with nixpkgs' toolchain
              pkgs.rust-analyzer-unwrapped
              # Finally the toolchain
              toolchain
            ];
            packages = [
              pkgs.solana-cli
              pkgs.cargo-zigbuild
            ];
            # Environment variables
            RUST_SRC_PATH = "${toolchain}/lib/rustlib/src/rust/library";
            RUSTFLAGS = "-C target-cpu=native";
          };
      });
}
