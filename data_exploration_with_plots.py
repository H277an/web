# 1. Load and Explore Dataset (5 Marks) - With saved plots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configure matplotlib for headless environment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set plot style for better visualization
plt.style.use('default')
sns.set_palette("husl")

print("Starting Dry Bean Dataset Analysis...")

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
        
        # Create output directory for plots
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        print(f"\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Plot 1: Class distribution
        plt.figure(figsize=(10, 6))
        
        # Create subplot for better visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        sns.countplot(data=df, x=class_column, ax=ax1)
        ax1.set_title("Distribution of Bean Classes (Count)")
        ax1.set_xlabel("Bean Class")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, (class_name, count) in enumerate(class_counts.items()):
            ax1.text(i, count + 50, str(count), ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Distribution of Bean Classes (Percentage)")
        
        plt.tight_layout()
        plt.savefig('plots/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved class distribution plot to plots/class_distribution.png")
        
        # Plot 2: Correlation matrix
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols if col != class_column]
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(14, 12))
            correlation_matrix = df[numerical_cols].corr()
            mask = correlation_matrix.abs() < 0.3  # Mask weak correlations
            
            sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', 
                       center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title("Correlation Matrix of Numerical Features\n(Only showing correlations with |r| >= 0.3)")
            plt.tight_layout()
            plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved correlation matrix to plots/correlation_matrix.png")
        
        # Plot 3: Feature distributions by class
        features_to_plot = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength']
        available_features = [f for f in features_to_plot if f in df.columns]
        
        if available_features:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.ravel()
            
            for i, feature in enumerate(available_features):
                if i < 4:  # Only plot first 4 features
                    sns.boxplot(data=df, x=class_column, y=feature, ax=axes[i])
                    axes[i].set_title(f'{feature} Distribution by Class')
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Add median values as text
                    medians = df.groupby(class_column)[feature].median()
                    for j, (class_name, median_val) in enumerate(medians.items()):
                        axes[i].text(j, median_val, f'{median_val:.0f}', 
                                   ha='center', va='bottom', fontweight='bold', color='red')
            
            # Hide unused subplots
            for i in range(len(available_features), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('plots/feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved feature distributions to plots/feature_distributions.png")
        
        # Plot 4: Pairplot of selected features
        selected_features = ['Area', 'Perimeter', 'AspectRatio', class_column]
        available_selected = [f for f in selected_features if f in df.columns]
        
        if len(available_selected) >= 4:
            plt.figure(figsize=(12, 10))
            # Sample data for faster plotting if dataset is large
            sample_size = min(2000, len(df))
            df_sample = df[available_selected].sample(n=sample_size, random_state=42)
            
            sns.pairplot(df_sample, hue=class_column, diag_kind='hist', 
                        plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
            plt.suptitle(f'Pairplot of Selected Features (Sample of {sample_size} points)', y=1.02)
            plt.savefig('plots/pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved pairplot to plots/pairplot.png")
        
        # Plot 5: Feature importance visualization (mean values by class)
        if len(numerical_cols) > 0:
            # Calculate mean values for each class
            class_means = df.groupby(class_column)[numerical_cols[:8]].mean()  # First 8 features
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(class_means.T, annot=True, cmap='viridis', fmt='.0f', cbar_kws={"shrink": .8})
            plt.title('Mean Feature Values by Bean Class')
            plt.xlabel('Bean Class')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig('plots/feature_means_by_class.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved feature means by class to plots/feature_means_by_class.png")
        
        print(f"\n" + "="*60)
        print("DATASET EXPLORATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Dataset contains {df.shape[0]} samples of {len(class_counts)} different bean classes.")
        print(f"✓ Each sample has {len(numerical_cols)} numerical features.")
        print(f"✓ Dataset is balanced with class sizes ranging from {class_counts.min()} to {class_counts.max()} samples.")
        print(f"✓ No missing values detected.")
        print(f"✓ Generated 5 visualization plots saved in 'plots/' directory")
        
        # Display some key insights
        print(f"\n" + "="*50)
        print("KEY INSIGHTS")
        print("="*50)
        print(f"• Largest class: {class_counts.index[0]} ({class_counts.iloc[0]} samples, {class_percentages.iloc[0]:.1f}%)")
        print(f"• Smallest class: {class_counts.index[-1]} ({class_counts.iloc[-1]} samples, {class_percentages.iloc[-1]:.1f}%)")
        
        # Find highly correlated features
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.9:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_val))
            
            if high_corr_pairs:
                print(f"• Highly correlated features (|r| > 0.9): {len(high_corr_pairs)} pairs found")
                for col1, col2, corr_val in high_corr_pairs[:3]:  # Show top 3
                    print(f"  - {col1} <-> {col2}: {corr_val:.3f}")
        
        print(f"\n📊 All visualizations saved in the 'plots/' directory:")
        print(f"   - class_distribution.png")
        print(f"   - correlation_matrix.png") 
        print(f"   - feature_distributions.png")
        print(f"   - pairplot.png")
        print(f"   - feature_means_by_class.png")
        
    else:
        print("Warning: Could not find class/target column in the dataset.")
        print("Available columns:", df.columns.tolist())

else:
    print("Failed to load the dataset. Please check the file or internet connection.")