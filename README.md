# Dry Bean Dataset Analysis

## Overview
This project implements the "Load and Explore Dataset" component (5 Marks) for analyzing the Dry Bean Dataset from the UCI Machine Learning Repository. The dataset contains 13,611 samples of 7 different registered dry bean varieties with 16 numerical features extracted from high-resolution camera images.

## Dataset Information
- **Source**: UCI Machine Learning Repository (ID: 602)
- **Samples**: 13,611 bean grains
- **Features**: 16 numerical features (12 dimensions + 4 shape forms)
- **Classes**: 7 different bean varieties
- **Missing Values**: None

### Bean Classes
1. **DERMASON** - 3,546 samples (26.1%)
2. **SIRA** - 2,636 samples (19.4%)
3. **SEKER** - 2,027 samples (14.9%)
4. **HOROZ** - 1,928 samples (14.2%)
5. **CALI** - 1,630 samples (12.0%)
6. **BARBUNYA** - 1,322 samples (9.7%)
7. **BOMBAY** - 522 samples (3.8%)

### Features
1. **Area** - Number of pixels within bean boundaries
2. **Perimeter** - Bean circumference length
3. **MajorAxisLength** - Distance between ends of longest line
4. **MinorAxisLength** - Longest perpendicular line to major axis
5. **AspectRatio** - Relationship between major and minor axis
6. **Eccentricity** - Eccentricity of equivalent ellipse
7. **ConvexArea** - Pixels in smallest convex polygon containing bean
8. **EquivDiameter** - Diameter of circle with same area
9. **Extent** - Ratio of pixels in bounding box to bean area
10. **Solidity** - Convexity ratio
11. **Roundness** - Calculated as (4πA)/P²
12. **Compactness** - Ed/L ratio
13. **ShapeFactor1-4** - Additional shape descriptors

## Files Included

### Python Scripts
1. **`data_exploration.py`** - Original implementation with error handling
2. **`data_exploration_uci.py`** - Enhanced version with UCI ML Repo integration
3. **`data_exploration_simple.py`** - Simplified version for headless environments
4. **`data_exploration_with_plots.py`** - Complete analysis with saved visualizations

### Jupyter Notebook
- **`dry_bean_exploration.ipynb`** - Interactive notebook version

### Generated Visualizations (in `plots/` directory)
1. **`class_distribution.png`** - Bean class count and percentage distributions
2. **`correlation_matrix.png`** - Feature correlation heatmap
3. **`feature_distributions.png`** - Box plots of key features by class
4. **`pairplot.png`** - Pairwise feature relationships
5. **`feature_means_by_class.png`** - Mean feature values by bean class

## Key Findings

### Dataset Quality
- ✅ **Complete Dataset**: No missing values detected
- ✅ **Sufficient Size**: 13,611 samples provide robust statistical power
- ✅ **Feature Rich**: 16 comprehensive morphological features

### Class Distribution
- **Moderate Imbalance**: Ranges from 522 (BOMBAY) to 3,546 (DERMASON) samples
- **Largest Class**: DERMASON beans (26.1% of dataset)
- **Smallest Class**: BOMBAY beans (3.8% of dataset)

### Feature Correlations
- **21 highly correlated pairs** (|r| > 0.9) identified
- Strong correlations between size-related features:
  - Area ↔ Perimeter (r = 0.967)
  - Area ↔ ConvexArea (r = 1.000)
  - Area ↔ MinorAxisLength (r = 0.952)

### Class Separability
- Different bean classes show distinct feature distributions
- **BOMBAY** beans are significantly larger than other varieties
- **DERMASON** beans are among the smallest
- **HOROZ** beans have high aspect ratios (elongated shape)

## Usage Instructions

### Basic Analysis
```bash
python3 data_exploration_simple.py
```

### Complete Analysis with Visualizations
```bash
python3 data_exploration_with_plots.py
```

### Jupyter Notebook
```bash
jupyter notebook dry_bean_exploration.ipynb
```

## Requirements
- Python 3.7+
- pandas
- seaborn
- matplotlib
- openpyxl (for Excel file support)
- ucimlrepo (for UCI ML Repository access)

### Installation
```bash
# System packages (Ubuntu/Debian)
sudo apt install python3-pandas python3-seaborn python3-matplotlib python3-openpyxl

# Or using pip
pip install pandas seaborn matplotlib openpyxl ucimlrepo
```

## Output Summary

The analysis successfully:
1. ✅ Loaded the Dry Bean Dataset (13,611 samples, 17 columns)
2. ✅ Displayed first 5 rows and basic statistics
3. ✅ Verified data quality (no missing values)
4. ✅ Analyzed class distribution across 7 bean varieties
5. ✅ Generated comprehensive visualizations
6. ✅ Identified key feature correlations and class characteristics

## Next Steps
This exploration provides the foundation for:
- Feature selection and engineering
- Machine learning model development
- Classification algorithm comparison
- Dimensionality reduction analysis

## Citation
```
Koklu, M. & Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture, 174, 105507.
```

## Dataset Reference
Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B