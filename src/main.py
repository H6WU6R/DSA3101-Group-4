import os
import subprocess
from A1_Customer_Segmentation.A1_main import main as A1_main
from A2_Customer_Engagement.A2_main import main as A2_main
from A3_Behavioral_Patterns_Analysis.A3_main import main as A3_main
from A4_Campaign_Impact_Analysis.A4_main import main as A4_main
from A5_Segmentation_Updates.app import main as A5_app
from B1_Predicting_Customer_Preferences.B1_main import main as B1_main
from B3_Measuring_Campaign_ROI.B3_main import main as B3_main
from B4_Cost_Effectiveness_of_Campaigns.B4_main import main as B4_main
from B5_Customer_Retention_Strategies.B5_main import main as B5_main

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

if __name__ == "__main__":
    # Step 1: Decrypt raw data files in the data/raw folder
    print("---------- Decrypting Raw Data ----------")
    # For demo purposes, we hardcode the passphrase.
    # In production, consider using an environment variable.
    PASS = "Group4"
    decrypt_raw_data(PASS)
    print("Raw data decryption complete.\n")

    print("-------------------------- Running A1: Customer Segmentation --------------------------\n")
    A1_main()
    
    print("-------------------------- Running A2: Customer Engagement --------------------------\n")
    A2_main()
    
    print("-------------------------- Running A3: Behavioral Patterns --------------------------\n")
    A3_main()
    
    print("-------------------------- Running A4: Campaign Impact Analysis --------------------------\n")
    A4_main()
    
    print("-------------------------- Running A5: Segmentation Updates --------------------------\n")
    A5_app()

    print("-------------------------- Running B1: Predicting Customer Preferences --------------------------\n")
    B1_main()
    
    print("-------------------------- Refer to markdown for Task B2: Campaign Optimization --------------------------\n")
    
    print("-------------------------- Running B3: Measuring Campaign ROI --------------------------\n")
    B3_main()
    
    print("-------------------------- Running B4: Cost Effectiveness of Campaigns --------------------------\n")
    B4_main()
    
    print("-------------------------- Running B5: Customer Retention Strategies --------------------------\n")
    B5_main()
    
    print("\nAll modules executed successfully!")
