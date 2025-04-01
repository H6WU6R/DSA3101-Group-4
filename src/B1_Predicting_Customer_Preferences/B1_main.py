from B1_Predicting_Customer_Preferences.dataclean import main as clean_main
from B1_Predicting_Customer_Preferences.data_Imputation_and_label_construction import main as label_main
from B1_Predicting_Customer_Preferences.trainmodel import main as train_main


def main():
    print("----- Running data cleaning for recommendation system -----\n")
    clean_main()
    
    print("----- Running data imputation and label construction for recommendation system -----\n")
    label_main()
    
    print("----- Running model training for recommendation system -----\n")
    train_main()
    
    
    print("\nAll recommendation functions executed successfully!")

if __name__ == "__main__":
    main()