# 1. Load and Explore Dataset (5 Marks)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Check if dataset file exists
dataset_file = "Dry_Bean_Dataset.xlsx"
if not os.path.exists(dataset_file):
    print(f"Dataset file '{dataset_file}' not found in the current directory.")
    print("Please ensure the Dry_Bean_Dataset.xlsx file is in the workspace.")
    print("You can download it from: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset")
    exit(1)

try:
    # Load Excel file
    df = pd.read_excel("Dry_Bean_Dataset.xlsx", sheet_name="Dry_Beans_Dataset")
    
    # Display first 5 rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Basic EDA
    print("\nDataset Shape:", df.shape)
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Data types information
    print("\nData Types:")
    print(df.dtypes)
    
    # Check unique values in Class column
    if 'Class' in df.columns:
        print("\nUnique Classes:")
        print(df['Class'].value_counts())
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    if 'Class' in df.columns:
        sns.countplot(data=df, x='Class')
        plt.title("Distribution of Bean Classes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'Class' not found in the dataset. Available columns:")
        print(df.columns.tolist())

except FileNotFoundError:
    print(f"Error: Could not find the file '{dataset_file}'")
    print("Please make sure the file is in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")