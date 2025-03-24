from web3 import Web3  # Import Web3 from web3.py

# Define blockchain provider URL (Infura Sepolia Testnet)
BLOCKCHAIN_PROVIDER_URL = "https://sepolia.infura.io/v3/2baecf4884314ac4ba5bc178092bc828"

# Initialize Web3 connection
w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_PROVIDER_URL))

# Check connection status
if w3.is_connected():
    print("Connected to Sepolia testnet")
else:
    print("Connection failed")
