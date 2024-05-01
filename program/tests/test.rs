use solana_program::hash::Hash;
use solana_program_test::{processor, BanksClient, ProgramTest};
use solana_sdk::{
    compute_budget::ComputeBudgetInstruction,
    signature::{Keypair, Signer},
    transaction::Transaction,
};

#[tokio::test]
async fn test_initialize() {
    // Setup
    let (mut banks, payer, blockhash) = setup_program_test_env().await;

    // Should fail
    let tx = build_tx(&payer, 0, 4, blockhash);
    assert!(banks.process_transaction(tx).await.is_err());

    // Should succeed
    let tx = build_tx(&payer, 6, 4, blockhash);
    assert!(banks.process_transaction(tx).await.is_ok());
}

fn build_tx(payer: &Keypair, nonce: u64, difficulty: u64, blockhash: Hash) -> Transaction {
    let cu_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(1_400_000);
    let ix = program::verify(payer.pubkey(), nonce, difficulty);
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
