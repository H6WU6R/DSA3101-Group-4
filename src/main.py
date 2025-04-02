print("-------------------------- importing modules... --------------------------\n")
from A1_Customer_Segmentation.A1_main import main as A1_main
from A2_Customer_Engagement.A2_main import main as A2_main
from A3_Behavioral_Patterns_Analysis.A3_main import main as A3_main
from A4_Campaign_Impact_Analysis.A4_main import main as A4_main
from A5_Segmentation_Updates.app import main as A5_app
from B1_Predicting_Customer_Preferences.B1_main import main as B1_main
from B3_Campaign_ROI_Evaluation.B3_main import main as B3_main
from B4_Cost_Effectiveness_Of_Campaigns.B4_main import main as B4_main
from B5_Customer_Retention_Strategies.B5_main import main as B5_main


if __name__ == "__main__":

    print("-------------------------- Running A1: Customer Segmentation --------------------------\n")
    A1_main()
    
    print("-------------------------- Running A2: Customer Engagement --------------------------\n")
    A2_main()
    
    print("-------------------------- Running A3: Behavioral Patterns --------------------------\n")
    A3_main()
    
    print("-------------------------- Running A4: Campaign Impact Analysis --------------------------\n")
    A4_main()
    
    print("-------------------------- Running B1: Predicting Customer Preferences --------------------------\n")
    B1_main()
    
    print("---------------------- Refer to markdown for Task B2: Campaign Optimization ----------------------\n")
    
    print("-------------------------- Running B3: Measuring Campaign ROI --------------------------\n")
    B3_main()
    
    print("-------------------------- Running B4: Cost Effectiveness of Campaigns --------------------------\n")
    B4_main()
    
    print("-------------------------- Running B5: Customer Retention Strategies --------------------------\n")
    B5_main()
    
    print("--------------------------All backend modules executed successfully!--------------------------")
    print("\n")
    print("----------------------------- Running Web App -----------------------------\n")
    print("-------------------------- Running A5: Segmentation Updates --------------------------\n")
    print("-------------------- Please Press CMD/CTRL + Click The Localhost To Open --------------------\n")
    print("-------------------- Or Copy the Localhost And Paste In Browser --------------------\n")
    print("\n")
    A5_app()
