use drillx::Solution;
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

    // Hash
    let challenge = [255; 32];
    let nonce = 0u64;
    let hash = drillx::hash(&challenge, &nonce.to_le_bytes()).unwrap();

    // Should succeed
    let s = Solution::new(hash.d, nonce.to_le_bytes());
    let tx = build_tx(&payer, 0, s, blockhash);
    assert!(banks.process_transaction(tx).await.is_ok());

    // Should fail
    let s = Solution::new(hash.d, nonce.to_le_bytes());
    let tx = build_tx(&payer, 24, s, blockhash);
    assert!(banks.process_transaction(tx).await.is_err());

    // Should fail
    let bad_nonce = 1u64;
    let s = Solution::new(hash.d, bad_nonce.to_le_bytes());
    let tx = build_tx(&payer, 0, s, blockhash);
    assert!(banks.process_transaction(tx).await.is_err());
}

fn build_tx(payer: &Keypair, difficulty: u64, solution: Solution, blockhash: Hash) -> Transaction {
    let cu_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(1_400_000);
    let ix = program::verify(payer.pubkey(), difficulty, solution.n, solution.d);
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
