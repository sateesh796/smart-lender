import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = "/Users/sarayu/Downloads/loan_prediction.csv "
OUTPUT_DIR = "bivariate_output"
TARGET_COL = "Loan_Status"

def run_bivariate_analysis():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    # Drop ID columns that don't add analytical value
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    features = [col for col in df.columns if col != TARGET_COL]

    for col in features:
        plt.figure(figsize=(8, 5))
        
        # Check if feature is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Boxplot for Numeric vs Categorical Target
            sns.boxplot(x=TARGET_COL, y=col, data=df)
            plt.title(f"Bivariate: {col} vs {TARGET_COL}")
        else:
            # Countplot for Categorical vs Categorical Target
            sns.countplot(x=col, hue=TARGET_COL, data=df)
            plt.title(f"Bivariate: {col} vs {TARGET_COL}")
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"bivariate_{col}.png"))
        plt.close()
        print(f"Generated plot for {col}")

    print(f"Bivariate analysis complete! Check the folder '{OUTPUT_DIR}'")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    run_bivariate_analysis()