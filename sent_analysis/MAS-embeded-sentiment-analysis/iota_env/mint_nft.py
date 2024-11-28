import os
from dotenv import load_dotenv
from iota_sdk import MintNftParams, Wallet, utf8_to_hex

# Load environment variables
load_dotenv()

def mint_nft(sentiment, text):
    """Mints an NFT with dynamic metadata based on sentiment and user input."""
    try:
        # Initialize the wallet
        wallet = Wallet(os.environ['WALLET_DB_PATH'])

        # Ensure the stronghold password is set
        if 'STRONGHOLD_PASSWORD' not in os.environ:
            raise Exception(".env STRONGHOLD_PASSWORD is undefined, see .env.example")

        wallet.set_stronghold_password(os.environ["STRONGHOLD_PASSWORD"])

        # Access the account
        account = wallet.get_account('Alice')

        # Sync the account with the node
        account.sync()

        # Create dynamic metadata
        metadata = f"some immutable nft metadata for {sentiment} on {text}"
        outputs = [MintNftParams(
            immutableMetadata=utf8_to_hex(metadata),
        )]

        # Mint the NFT
        transaction = account.mint_nfts(outputs)
        print(f'NFT minted successfully! Block sent: {os.environ["EXPLORER_URL"]}/block/{transaction.blockId}')
    except Exception as e:
        print(f"Error minting NFT: {e}")