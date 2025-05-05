use bytemuck::{Pod, Zeroable};
use solana_program::{
    self, account_info::AccountInfo, declare_id, entrypoint::ProgramResult,
    instruction::Instruction, program_error::ProgramError, pubkey::Pubkey,
};

declare_id!("DV1J1tBiRCSs8czHAJT449nP569c7eCTHQK4sK9NeWRP");

#[cfg(not(feature = "no-entrypoint"))]
solana_program::entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo],
    data: &[u8],
) -> ProgramResult {
    let args = Args::try_from_bytes(data)?;

    let challenge = args.challenge;
    let nonce = args.nonce;
    let solution = args.solution;

    let expected = drillx::hash(&challenge, &nonce);

    solana_program::log::sol_log(&format!("expected: {:?}", expected));
    solana_program::log::sol_log(&format!("solution: {:?}", solution));

    if expected != solution {
        return Err(ProgramError::InvalidAccountData);
    }

    Ok(())
}

pub fn verify(
    signer: Pubkey,
    challenge: [u8; 32],
    nonce: [u8; 8],
    solution: [u8; 32],
) -> Instruction {
    Instruction {
        program_id: crate::id(),
        accounts: vec![],
        data: Args {
            challenge,
            nonce,
            solution,
        }
        .to_bytes()
        .to_vec(),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Args {
    pub challenge: [u8; 32],
    pub nonce: [u8; 8],
    pub solution: [u8; 32],
}

impl Args {
    pub fn to_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    fn try_from_bytes(data: &[u8]) -> Result<&Self, ProgramError> {
        bytemuck::try_from_bytes::<Self>(&data).or(Err(ProgramError::InvalidAccountData))
    }
}
