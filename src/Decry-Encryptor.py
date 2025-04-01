import os
import subprocess
import pandas as pd
from cryptography.fernet import Fernet

def encrypt_text(text, key):
    """
    Encrypts a text string using the provided Fernet key.
    Returns the encrypted text as a string.
    """
    fernet = Fernet(key)
    return fernet.encrypt(text.encode()).decode()

def encrypt_sensitive_columns(df, columns, key):
    """
    Encrypts the specified columns in the DataFrame using Fernet encryption.
    Formats numeric data consistently before encrypting.
    """
    df = df.copy()
    for col in columns:
        # Format numeric columns consistently
        if col == 'card_number':
            df[col] = df[col].astype(str).str.zfill(16)  # Ensure 16 digits
        elif col == 'cvv':
            df[col] = df[col].astype(str).str.zfill(3)   # Ensure 3 digits
        # Encrypt each cell in the column
        df[col] = df[col].apply(lambda x: encrypt_text(x, key))
    return df

def decrypt_and_encrypt_sensitive_data(passphrase, encryption_key, data_dir="data/raw", sensitive_columns=None):
    """
    Decrypts all .gpg files in data_dir using the given GPG passphrase.
    If a decrypted file is a CSV and contains any of the sensitive columns,
    the function encrypts those columns and overwrites the decrypted CSV with the encrypted data.
    If no sensitive columns are found, the decrypted file is left as is.
    """
    if sensitive_columns is None:
        sensitive_columns = ['card_number', 'cvv']
    
    # Process each .gpg file in the specified directory.
    for filename in os.listdir(data_dir):
        if filename.endswith(".gpg"):
            encrypted_file = os.path.join(data_dir, filename)
            decrypted_file = os.path.join(data_dir, filename[:-4])  # Remove the .gpg extension
            print(f"Decrypting {encrypted_file} to {decrypted_file}...")
            
            # Build the GPG decryption command.
            command = [
                "gpg",
                "--batch",
                "--yes",
                "--decrypt",
                "--passphrase", passphrase,
                "-o", decrypted_file,
                encrypted_file
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error decrypting {encrypted_file}:\n{result.stderr}")
                continue
            else:
                print(f"Decrypted {encrypted_file} successfully.")
            
            # Process CSV files for sensitive data.
            if decrypted_file.endswith(".csv"):
                try:
                    df = pd.read_csv(decrypted_file)
                    print(f"Loaded CSV file: {decrypted_file}")
                except Exception as e:
                    print(f"Error reading {decrypted_file} as CSV: {e}")
                    continue
                
                # Identify sensitive columns present in the DataFrame.
                present_sensitive_columns = [col for col in sensitive_columns if col in df.columns]
                if present_sensitive_columns:
                    print(f"Sensitive columns {present_sensitive_columns} found in {decrypted_file}. Encrypting them...")
                    df_encrypted = encrypt_sensitive_columns(df, present_sensitive_columns, encryption_key)
                    # Overwrite the decrypted CSV with the encrypted version.
                    df_encrypted.to_csv(decrypted_file, index=False)
                    print(f"Encrypted data saved to {decrypted_file}")
                else:
                    print(f"No sensitive columns found in {decrypted_file}. Keeping unencrypted data.")

# GPG passphrase for decryption
gpg_passphrase = "Group4"
    
# Fernet encryption key for encrypting sensitive columns.
# In practice you might load this from a file or environment variable.
encryption_key = Fernet.generate_key()
    
# The directory containing your encrypted .gpg files
data_directory = "data/raw"
    
decrypt_and_encrypt_sensitive_data(gpg_passphrase, encryption_key, data_directory)
