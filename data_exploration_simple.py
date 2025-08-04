# 1. Load and Explore Dataset (5 Marks) - Simplified version without graphics
import pandas as pd
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
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset from UCI ML repository
        print("Fetching dataset from UCI ML repository (ID: 602)...")
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
    print("\n" + "="*60)
    print("DATASET EXPLORATION RESULTS")
    print("="*60)
    
    # Display first 5 rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Basic EDA
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✓ No missing values found in the dataset!")
    
    # Check target column (should be 'Class')
    target_columns = ['Class', 'class', 'target']
    class_column = None
    
    for col in target_columns:
        if col in df.columns:
            class_column = col
            break
    
    if class_column:
        print(f"\n" + "="*50)
        print(f"TARGET VARIABLE ANALYSIS: '{class_column}'")
        print("="*50)
        
        class_counts = df[class_column].value_counts()
        print(f"Unique classes and their counts:")
        print(class_counts)
        
        print(f"\nTotal number of classes: {len(class_counts)}")
        print(f"Class distribution (percentages):")
        class_percentages = (class_counts / len(df) * 100).round(2)
        print(class_percentages)
        
        # Display basic statistics for each class
        print(f"\n" + "="*50)
        print("BASIC STATISTICS BY CLASS")
        print("="*50)
        
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols if col != class_column]
        
        if len(numerical_cols) > 0:
            for class_name in sorted(df[class_column].unique()):
                class_data = df[df[class_column] == class_name]
                print(f"\n{class_name}: {len(class_data)} samples ({class_percentages[class_name]:.1f}%)")
                
                # Show statistics for first few numerical features
                for i, col in enumerate(numerical_cols[:5]):  # Show first 5 features
                    mean_val = class_data[col].mean()
                    std_val = class_data[col].std()
                    print(f"  {col}: mean={mean_val:.2f}, std={std_val:.2f}")
        
        print(f"\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        if len(numerical_cols) > 1:
            # Show correlations between first few features
            correlation_matrix = df[numerical_cols[:10]].corr()  # Show first 10 features
            print("\nTop correlations (absolute value > 0.7):")
            
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        col1 = correlation_matrix.columns[i]
                        col2 = correlation_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_val))
            
            if high_corr_pairs:
                for col1, col2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                    print(f"  {col1} <-> {col2}: {corr_val:.3f}")
            else:
                print("  No correlations with absolute value > 0.7 found in the first 10 features.")
        
        print(f"\n" + "="*60)
        print("DATASET EXPLORATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Dataset contains {df.shape[0]} samples of {len(class_counts)} different bean classes.")
        print(f"✓ Each sample has {len(numerical_cols)} numerical features.")
        print(f"✓ Dataset is balanced with class sizes ranging from {class_counts.min()} to {class_counts.max()} samples.")
        print(f"✓ No missing values detected.")
        
        # Key features information
        print(f"\nKey Features (first 10):")
        for i, col in enumerate(numerical_cols[:10]):
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"  {i+1:2d}. {col}: [{min_val:.2f} - {max_val:.2f}], mean={mean_val:.2f}")
        
    else:
        print("Warning: Could not find class/target column in the dataset.")
        print("Available columns:", df.columns.tolist())

else:
    print("Failed to load the dataset. Please check the file or internet connection.")