from dataclean import main as clean_main
from data_Imputation_and_label_construction import main as label_main
from trainmodel import main as train_main


if __name__ == "__main__":
    print("----- Running data cleaning for recommendation system -----\n")
    clean_main()
    
    print("----- Running data imputation and label construction for recommendation system -----\n")
    label_main()
    
    print("----- Running model training for recommendation system -----\n")
    train_main()
    
    
    print("\nAll recommendation functions executed successfully!")
