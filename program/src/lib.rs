use std::ops::Add;

use bytemuck::{Pod, Zeroable};
use num_bigint::BigInt;
use num_traits::{FromBytes, ToPrimitive};
use solana_program::{
    self,
    account_info::AccountInfo,
    declare_id,
    entrypoint::ProgramResult,
    instruction::{AccountMeta, Instruction},
    keccak::hashv,
    program_error::ProgramError,
    pubkey::Pubkey,
};

declare_id!("mineRHF5r6S7HyD9SppBfVMXMavDkJsxwGesEvxZr2A");

#[cfg(not(feature = "no-entrypoint"))]
solana_program::entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    data: &[u8],
) -> ProgramResult {
    let args = Args::try_from_bytes(data)?;

    let [_signer, noise] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };

    let challenge = [255; 32];
    let candidate = drill_hash(challenge, args.nonce, &noise.data.borrow());
    if difficulty(candidate).lt(&args.difficulty) {
        return Err(ProgramError::Custom(0));
    }

    Ok(())
}

fn drill_hash(challenge: [u8; 32], nonce: u64, noise: &[u8]) -> [u8; 32] {
    // The drill part (1024 sequential modpow and mem reads)
    let len = BigInt::from(noise.len());
    let mut digest = [0u8; 1024];
    let mut addr = BigInt::from_le_bytes(&challenge);
    let mut n = BigInt::from(nonce);
    for i in 0..1024 {
        // TODO Handle nonce = 0 and 1 case better
        addr = addr.modpow(&n.add(2), &len);
        digest[i] = noise[addr.to_usize().unwrap()];
        n = BigInt::from(digest[i]);
    }

    // The hash part (keccak proof)
    hashv(&[digest.as_slice()]).to_bytes()
}

fn difficulty(hash: [u8; 32]) -> u64 {
    let mut count = 0;
    for &byte in &hash {
        let lz = byte.leading_zeros();
        count += lz;
        if lz < 8 {
            break;
        }
    }
    count.into()
}

pub fn verify(signer: Pubkey, nonce: u64, difficulty: u64) -> Instruction {
    Instruction {
        program_id: crate::id(),
        accounts: vec![
            AccountMeta::new(signer, true),
            AccountMeta::new_readonly(noise_address(), false),
        ],
        data: Args { nonce, difficulty }.to_bytes().to_vec(),
    }
}

pub const NOISE: &str = "noise";

pub fn noise_address() -> Pubkey {
    Pubkey::find_program_address(&[NOISE.as_ref()], &crate::id()).0
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Args {
    pub difficulty: u64,
    pub nonce: u64,
}

impl Args {
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    fn try_from_bytes(data: &[u8]) -> Result<&Self, ProgramError> {
        bytemuck::try_from_bytes::<Self>(&data).or(Err(ProgramError::InvalidAccountData))
    }
}
