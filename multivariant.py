import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plti
DATA_PATH = "/content/loan_prediction.csv "
OUTPUT_DIR = "multivariate_output"
TARGET_COL = "Loan_Status"

def run_multivariate_analysis():
    # 1. Setup
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    # 2. Prepare Data
    # Drop IDs and handle missing values for clean visualization
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
    df_clean = df.dropna()
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    print("Generating Multivariate Visualizations...")

    # 3. Pairplot (Interactions between multiple numeric features)
    # This shows how multiple numeric variables relate to each other, colored by target
    if len(numeric_cols) > 1:
        print("Creating Pairplot...")
        pairplot = sns.pairplot(df_clean, hue=TARGET_COL, palette="viridis", corner=True)
        pairplot.fig.suptitle("Multivariate Interaction: Pairplot", y=1.02)
        plt.savefig(os.path.join(OUTPUT_DIR, "multivariate_pairplot.png"), dpi=100)
        plt.close()

    # 4. Clustered Heatmap
    # This groups features that behave similarly, revealing hidden structures
    if len(numeric_cols) > 2:
        print("Creating Clustered Heatmap...")
        corr = df[numeric_cols].corr()
        clustermap = sns.clustermap(corr, annot=True, cmap="vlag", figsize=(10, 10))
        clustermap.fig.suptitle("Multivariate Structure: Clustered Heatmap", y=1.02)
        plt.savefig(os.path.join(OUTPUT_DIR, "multivariate_clustermap.png"), dpi=150)
        plt.close()

    print(f"Multivariate analysis complete! Check the folder '{OUTPUT_DIR}'")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    run_multivariate_analysis()