import os
import subprocess

def decrypt_raw_data(passphrase):
    """
    Decrypts all .gpg files in the data/raw folder using the given passphrase.
    The decrypted files are saved in the same folder with the .gpg extension removed.
    """
    DATA_DIR = "data/raw"
    
    # List all files in the data/raw directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".gpg"):
            encrypted_file = os.path.join(DATA_DIR, filename)
            # Remove the .gpg extension for the decrypted file
            decrypted_file = os.path.join(DATA_DIR, filename[:-4])
            print(f"Decrypting {encrypted_file} to {decrypted_file}...")
            
            # Build the GPG command for decryption
            command = [
                "gpg",
                "--batch",
                "--yes",
                "--decrypt",
                "--passphrase", passphrase,
                "-o", decrypted_file,
                encrypted_file
            ]
            # Execute the command
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error decrypting {encrypted_file}:\n{result.stderr}")
            else:
                print(f"Decrypted {encrypted_file} successfully.")

                
# Step 1: Decrypt raw data files in the data/raw folder
print("---------- Decrypting Raw Data ----------")
# For demo purposes, we hardcode the passphrase.
# In production, consider using an environment variable.
PASS = "Group4"
decrypt_raw_data(PASS)
print("Raw data decryption complete.\n")
