import os
from dotenv import load_dotenv
from iota_sdk import SendParams, Wallet

# Load environment variables
load_dotenv()

def perform_transaction():
    try:
        # Initialize the wallet
        wallet = Wallet(os.environ['WALLET_DB_PATH'])

        # Access the account
        account = wallet.get_account('Alice')

        # Sync the account with the node
        account.sync()

        # Ensure the stronghold password is set
        if 'STRONGHOLD_PASSWORD' not in os.environ:
            raise Exception(".env STRONGHOLD_PASSWORD is undefined, see .env.example")

        wallet.set_stronghold_password(os.environ["STRONGHOLD_PASSWORD"])

        # Define the transaction parameters
        params = [SendParams(
            address="rms1qpszqzadsym6wpppd6z037dvlejmjuke7s24hm95s9fg9vpua7vluaw60xu",
            amount=100000,  # Amount in smallest units
        )]

        # Execute the transaction
        transaction = account.send_with_params(params)
        print(f'Transaction successful! Block sent: {os.environ["EXPLORER_URL"]}/block/{transaction.blockId}')
        return transaction.blockId
    except Exception as e:
        print(f"Error performing transaction: {e}")
        raise
