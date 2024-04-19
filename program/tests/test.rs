use drillhash::noise_address;
use solana_program::{hash::Hash, rent::Rent};
use solana_program_test::{processor, read_file, BanksClient, ProgramTest};
use solana_sdk::{
    account::Account,
    signature::{Keypair, Signer},
    transaction::Transaction,
};

#[tokio::test]
async fn test_initialize() {
    // Setup
    let (mut banks, payer, blockhash) = setup_program_test_env().await;

    // Should fail
    let ix = drillhash::verify(payer.pubkey(), 0, 4);
    let tx = Transaction::new_signed_with_payer(&[ix], Some(&payer.pubkey()), &[&payer], blockhash);
    assert!(banks.process_transaction(tx).await.is_err());

    // Should succeed
    let ix = drillhash::verify(payer.pubkey(), 1, 4);
    let tx = Transaction::new_signed_with_payer(&[ix], Some(&payer.pubkey()), &[&payer], blockhash);
    assert!(banks.process_transaction(tx).await.is_ok());
}

async fn setup_program_test_env() -> (BanksClient, Keypair, Hash) {
    let mut program_test = ProgramTest::new(
        "drillhash",
        drillhash::ID,
        processor!(drillhash::process_instruction),
    );
    program_test.prefer_bpf(true);

    // Setup metadata program
    let data = read_file(&"../noise.txt");
    program_test.add_account(
        noise_address(),
        Account {
            lamports: Rent::default().minimum_balance(data.len()).max(1),
            data,
            owner: drillhash::id(),
            executable: true,
            rent_epoch: 0,
        },
    );

    program_test.start().await
}
