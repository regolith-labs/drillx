use bytemuck::{Pod, Zeroable};
use solana_program::{
    self,
    account_info::AccountInfo,
    declare_id,
    entrypoint::ProgramResult,
    instruction::{AccountMeta, Instruction},
    log::sol_log_compute_units,
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

    let [_signer] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };

    let challenge = [255; 32];
    let candidate = drillx::hash(&challenge, &args.nonce);
    if drillx::difficulty(candidate).lt(&(args.difficulty as u32)) {
        return Err(ProgramError::Custom(0));
    }

    sol_log_compute_units();
    Err(ProgramError::Custom(0))
    // Ok(())
}

pub fn verify(signer: Pubkey, nonce: u64, difficulty: u64) -> Instruction {
    Instruction {
        program_id: crate::id(),
        accounts: vec![AccountMeta::new(signer, true)],
        data: Args {
            nonce: nonce.to_le_bytes(),
            difficulty,
        }
        .to_bytes()
        .to_vec(),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Args {
    pub nonce: [u8; 8],
    pub difficulty: u64,
}

impl Args {
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    fn try_from_bytes(data: &[u8]) -> Result<&Self, ProgramError> {
        bytemuck::try_from_bytes::<Self>(&data).or(Err(ProgramError::InvalidAccountData))
    }
}
