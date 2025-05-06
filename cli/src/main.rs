use drillx;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    compute_budget::ComputeBudgetInstruction,
    signature::{read_keypair_file, Signer},
    transaction::Transaction,
};

#[tokio::main]
async fn main() {
    // Read keypair from file
    let payer =
        read_keypair_file(&std::env::var("KEYPAIR").expect("Missing KEYPAIR env var")).unwrap();

    // Create test data
    let challenge = [255u8; 32];
    let nonce = 0u64.to_le_bytes();
    let solution = drillx::hash(&challenge, &nonce).unwrap();

    // Build transaction
    let rpc = RpcClient::new(std::env::var("RPC").expect("Missing RPC env var"));
    let cu_budget_ix = ComputeBudgetInstruction::set_compute_unit_limit(1_400_000);
    let verify_ix = program::verify(payer.pubkey(), challenge, solution.d, 0u64.to_le_bytes());
    let blockhash = rpc.get_latest_blockhash().await.unwrap();
    let transaction = Transaction::new_signed_with_payer(
        &[cu_budget_ix, verify_ix],
        Some(&payer.pubkey()),
        &[&payer],
        blockhash,
    );

    // Send transaction
    match rpc.send_and_confirm_transaction(&transaction).await {
        Ok(signature) => println!("Transaction succeeded! Signature: {}", signature),
        Err(err) => println!("Transaction failed: {}", err),
    }
}
