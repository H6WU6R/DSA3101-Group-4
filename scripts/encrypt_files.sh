#!/bin/bash
# encrypt_files.sh
# This script encrypts all files in the "data/" directory using GPG symmetric encryption.
# It does NOT delete the original plaintext file (for demonstration purposes).

# Usage Examples:
#   ./encrypt_files.sh
#   ./encrypt_files.sh "YourSecurePassphrase"

DATA_DIR="data"

# 1. Ensure the data directory exists.
if [ ! -d "$DATA_DIR" ]; then
  echo "Directory '$DATA_DIR' does not exist. Exiting."
  exit 1
fi

# 2. Get passphrase: from the first argument or prompt the user.
if [ $# -eq 0 ]; then
  read -sp "Enter passphrase for encryption: " PASSPHRASE
  echo ""
else
  PASSPHRASE="$1"
fi

# 3. Loop over files in the data directory.
for file in "$DATA_DIR"/*; do
  # Check if it's a regular file and not already encrypted
  if [ -f "$file" ] && [[ "$file" != *.gpg ]]; then
    echo "Encrypting $file..."
    # Pass the passphrase via standard input to GPG
    echo "$PASSPHRASE" | gpg --batch --yes \
      --symmetric --cipher-algo AES256 \
      --passphrase-fd 0 \
      "$file"

    if [ $? -eq 0 ]; then
      echo "Encryption successful: ${file}.gpg"
      # NOTE: We do NOT remove the original file here.
      # rm "$file"  # <-- commented out for demonstration
    else
      echo "Encryption failed for $file."
    fi
  fi
done

echo "Encryption process completed (original files remain)."
