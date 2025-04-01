#!/bin/bash
# decrypt_files.sh
# This script decrypts all .gpg files in the "data/" directory.
# Decrypted files are written alongside the original ones.
#
# Usage Examples:
#   ./decrypt_files.sh
#   ./decrypt_files.sh "YourSecurePassphrase"

DATA_DIR="data/raw"

# 1. Ensure the data directory exists.
if [ ! -d "$DATA_DIR" ]; then
  echo "Directory '$DATA_DIR' does not exist. Exiting."
  exit 1
fi

# 2. Get passphrase: from the first argument or prompt the user.
if [ $# -eq 0 ]; then
  read -sp "Enter passphrase for decryption: " PASSPHRASE
  echo ""
else
  PASSPHRASE="$1"
fi

# 3. Loop over all .gpg files in the data directory.
for file in "$DATA_DIR"/*.gpg; do
  if [ -f "$file" ]; then
    # Construct an output filename by removing the ".gpg" extension
    outfile="${file%.gpg}"
    echo "Decrypting $file to $outfile..."

    # Decrypt with the passphrase
    echo "$PASSPHRASE" | gpg --batch --yes \
      --decrypt --passphrase-fd 0 \
      -o "$outfile" \
      "$file"

    if [ $? -eq 0 ]; then
      echo "Decryption successful: $outfile"
      # NOTE: We do not delete the .gpg file. It's purely your choice.
    else
      echo "Decryption failed for $file."
    fi
  fi
done

echo "Decryption process completed."
