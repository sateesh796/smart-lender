import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/Users/sarayu/Downloads/loan_prediction.csv"
OUTPUT_DIR = "multivariate_output"
TARGET_COL = "Loan_Status"

def run_multivariate_analysis():
    # 1. Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}. Please check the path.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    # 2. Data Cleaning
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
    
    # Drop rows with missing values to prevent plotting errors
    df_clean = df.dropna()
    
    # Select ONLY numeric columns for multivariate plots
    numeric_df = df_clean.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for multivariate analysis.")
        return

    print(f"Generating visualizations for {numeric_df.shape[1]} numeric features...")

    # 3. Pairplot
    # We add back the target column for coloring
    df_plot = numeric_df.copy()
    df_plot[TARGET_COL] = df_clean[TARGET_COL]
    
    try:
        print("Creating Pairplot...")
        pairplot = sns.pairplot(df_plot, hue=TARGET_COL, palette="viridis", corner=True)
        pairplot.fig.suptitle("Multivariate Interaction: Pairplot", y=1.02)
        plt.savefig(os.path.join(OUTPUT_DIR, "multivariate_pairplot.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Pairplot error: {e}")

    # 4. Clustered Heatmap
    try:
        print("Creating Clustered Heatmap...")
        corr = numeric_df.corr()
        clustermap = sns.clustermap(corr, annot=True, cmap="vlag", figsize=(10, 10))
        clustermap.fig.suptitle("Multivariate Structure: Clustered Heatmap", y=1.02)
        plt.savefig(os.path.join(OUTPUT_DIR, "multivariate_clustermap.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Heatmap error: {e}")

    print(f"Multivariate analysis complete! Check the folder '{OUTPUT_DIR}'")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    run_multivariate_analysis()