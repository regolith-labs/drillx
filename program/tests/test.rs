use solana_program::hash::Hash;
use solana_program_test::{processor, BanksClient, ProgramTest};
use solana_sdk::{
    compute_budget::ComputeBudgetInstruction,
    signature::{Keypair, Signer},
    transaction::Transaction,
};

#[tokio::test]
async fn test_haraka() {
    // Setup
    let (mut banks, payer, blockhash) = setup_program_test_env().await;

    // Hash
    let challenge = [255; 32];
    let nonce = 0u64.to_le_bytes();
    let solution = drillx::hash(&challenge, &nonce);

    // Should succeed
    let tx = build_tx(&payer, challenge, nonce, solution, blockhash);
    assert!(banks.process_transaction(tx).await.is_ok());

    // Should fail
    // let tx = build_tx(&payer, challenge, 1u64.to_le_bytes(), solution, blockhash);
    // assert!(banks.process_transaction(tx).await.is_err());
}

fn build_tx(
    payer: &Keypair,
    challenge: [u8; 32],
    nonce: [u8; 8],
    solution: [u8; 32],
    blockhash: Hash,
) -> Transaction {
    let cu_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(1_400_000);
    let ix = program::verify(payer.pubkey(), challenge, nonce, solution);
    Transaction::new_signed_with_payer(
        &[cu_budget_ix, ix],
        Some(&payer.pubkey()),
        &[&payer],
        blockhash,
    )
}

async fn setup_program_test_env() -> (BanksClient, Keypair, Hash) {
    let mut program_test = ProgramTest::new(
        "program",
        program::id(),
        processor!(program::process_instruction),
    );
    program_test.prefer_bpf(true);
    program_test.start().await
}
