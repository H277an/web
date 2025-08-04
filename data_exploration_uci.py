# 1. Load and Explore Dataset (5 Marks) - Alternative approach using UCI ML Repo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Method 1: Try to load from local Excel file
dataset_file = "Dry_Bean_Dataset.xlsx"
df = None

if os.path.exists(dataset_file):
    print("Loading dataset from local Excel file...")
    try:
        df = pd.read_excel(dataset_file, sheet_name="Dry_Beans_Dataset")
        print("Successfully loaded from Excel file!")
    except Exception as e:
        print(f"Error loading Excel file: {e}")

# Method 2: If Excel file not available, try UCI ML repository
if df is None:
    print("Excel file not found. Attempting to download from UCI ML repository...")
    try:
        # Install ucimlrepo if not available
        import subprocess
        import sys
        
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            print("Installing ucimlrepo package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
            from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset from UCI ML repository
        dry_bean = fetch_ucirepo(id=602)
        
        # Combine features and target into a single DataFrame
        X = dry_bean.data.features
        y = dry_bean.data.targets
        df = pd.concat([X, y], axis=1)
        print("Successfully loaded from UCI ML repository!")
        
    except Exception as e:
        print(f"Error loading from UCI repository: {e}")
        print("Please manually download the dataset from:")
        print("https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset")
        exit(1)

if df is not None:
    # Display first 5 rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Basic EDA
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✓ No missing values found in the dataset!")
    
    # Data types information
    print("\nData Types:")
    print(df.dtypes)
    
    # Check target column (should be 'Class')
    target_columns = ['Class', 'class', 'target']
    class_column = None
    
    for col in target_columns:
        if col in df.columns:
            class_column = col
            break
    
    if class_column:
        print(f"\nTarget variable: '{class_column}'")
        print(f"Unique classes and their counts:")
        class_counts = df[class_column].value_counts()
        print(class_counts)
        
        print(f"\nTotal number of classes: {len(class_counts)}")
        print(f"Class distribution (percentages):")
        print((class_counts / len(df) * 100).round(2))
        
        # Plot class distribution
        plt.figure(figsize=(12, 8))
        
        # Create subplot for better visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        sns.countplot(data=df, x=class_column, ax=ax1)
        ax1.set_title("Distribution of Bean Classes (Count)")
        ax1.set_xlabel("Bean Class")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Distribution of Bean Classes (Percentage)")
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualizations
        print("\nGenerating correlation matrix for numerical features...")
        
        # Select only numerical columns (excluding the target)
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols if col != class_column]
        
        if len(numerical_cols) > 1:
            # Correlation matrix
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title("Correlation Matrix of Numerical Features")
            plt.tight_layout()
            plt.show()
            
            # Display basic statistics for each class
            print(f"\nBasic statistics by {class_column}:")
            for class_name in sorted(df[class_column].unique()):
                class_data = df[df[class_column] == class_name]
                print(f"\n{class_name}: {len(class_data)} samples")
                print(f"  Mean area: {class_data.get('Area', class_data.iloc[:, 0]).mean():.2f}")
                if 'Perimeter' in class_data.columns:
                    print(f"  Mean perimeter: {class_data['Perimeter'].mean():.2f}")
        
        print(f"\n✓ Dataset exploration completed successfully!")
        print(f"The dataset contains {df.shape[0]} samples of {len(class_counts)} different bean classes.")
        print(f"Each sample has {len(numerical_cols)} numerical features.")
        
    else:
        print("Warning: Could not find class/target column in the dataset.")
        print("Available columns:", df.columns.tolist())
        
        # Still show some basic visualizations
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 1:
            plt.figure(figsize=(10, 6))
            df[numerical_cols[:5]].hist(bins=20, figsize=(15, 10))
            plt.suptitle("Distribution of First 5 Numerical Features")
            plt.tight_layout()
            plt.show()

else:
    print("Failed to load the dataset. Please check the file or internet connection.")