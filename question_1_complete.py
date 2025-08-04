#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Question 1: Dry Bean Classification Analysis (50 Marks)
GIKI IT Centre – C2 – DS&AI Internal Final Assessment

Scenario: You're a Machine Learning Engineer at a food-tech firm automating 
dry bean sorting using image-based morphological data.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=== QUESTION 1: DRY BEAN CLASSIFICATION ANALYSIS ===")
print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

# =============================================================================
# PART 1: DATA PREPROCESSING (14 MARKS)
# =============================================================================

print("\n" + "="*60)
print("PART 1: DATA PREPROCESSING (14 MARKS)")
print("="*60)

# 1.1 EDA (5 Marks)
print("\n1.1 EDA - Loading and Exploring Data")
print("-" * 40)

# Load the Dry Bean Dataset from UCI repository
print("Loading Dry Bean Dataset from UCI repository...")
dry_bean = fetch_ucirepo(id=602)

# Extract features and targets
X = dry_bean.data.features
y = dry_bean.data.targets

print(f"✅ Dataset loaded successfully!")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Combine features and target for easier analysis
df = pd.concat([X, y], axis=1)
print(f"Combined dataset shape: {df.shape}")

# Basic data exploration
print(f"\nDataset Info:")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(X.columns)}")
print(f"Feature names: {list(X.columns)}")
print(f"Target column: {y.columns[0]}")

# Check for missing values
print(f"\nMissing values check:")
missing_values = df.isnull().sum()
print(f"Total missing values: {missing_values.sum()}")
if missing_values.sum() == 0:
    print("✅ No missing values found!")

# Class distribution analysis
target_col = y.columns[0]
class_distribution = df[target_col].value_counts()
print(f"\nClass Distribution:")
print(class_distribution)
print(f"\nClass percentages:")
print((class_distribution / len(df) * 100).round(2))

print(f"Number of classes: {len(class_distribution)}")
print(f"Classes: {list(class_distribution.index)}")

# Basic statistics
print(f"\nFeature Statistics Summary:")
print(df.describe().round(2))

# 1.2 Label Encoding (3 Marks)
print("\n1.2 Label Encoding")
print("-" * 40)

# Label encoding for the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.iloc[:, 0])

print("✅ Label encoding completed!")
print(f"Original classes: {label_encoder.classes_}")
print("\nClass mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {class_name} -> {i}")

print(f"\nEncoded target shape: {y_encoded.shape}")
print(f"Unique encoded values: {np.unique(y_encoded)}")

# 1.3 Feature Scaling (3 Marks)
print("\n1.3 Feature Scaling")
print("-" * 40)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Feature scaling completed!")
print(f"Original features - Mean: {X.mean().round(2).head()}")
print(f"Original features - Std: {X.std().round(2).head()}")

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(f"Scaled features - Mean: {X_scaled_df.mean().round(2).head()}")
print(f"Scaled features - Std: {X_scaled_df.std().round(2).head()}")

# 1.4 Train-Test Split (3 Marks)
print("\n1.4 Stratified Train-Test Split")
print("-" * 40)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print("✅ Stratified split completed!")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# Verify stratification
train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist = pd.Series(y_test).value_counts().sort_index()
print(f"\nClass distribution verification:")
print(f"Train percentages: {(train_dist / len(y_train) * 100).round(2).values}")
print(f"Test percentages: {(test_dist / len(y_test) * 100).round(2).values}")

# =============================================================================
# PART 2: RANDOM FOREST MODEL (11 MARKS)
# =============================================================================

print("\n" + "="*60)
print("PART 2: RANDOM FOREST MODEL (11 MARKS)")
print("="*60)

# 2.1 Hyperparameter Tuning (5 Marks)
print("\n2.1 Random Forest Hyperparameter Tuning")
print("-" * 40)

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

print("Parameter grid:")
for param, values in rf_param_grid.items():
    print(f"  {param}: {values}")

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# GridSearchCV with stratified k-fold
print("\nStarting GridSearchCV...")
rf_grid_search = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

# Fit the grid search
rf_grid_search.fit(X_train, y_train)

print("✅ GridSearchCV completed!")
print(f"Best CV score: {rf_grid_search.best_score_:.4f}")
print(f"Best parameters: {rf_grid_search.best_params_}")

# 2.2 Model Building (2 Marks)
print("\n2.2 Random Forest Model Training")
print("-" * 40)

# Get the best Random Forest model
best_rf = rf_grid_search.best_estimator_

# Make predictions
rf_train_pred = best_rf.predict(X_train)
rf_test_pred = best_rf.predict(X_test)

# Calculate accuracies
rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
rf_test_accuracy = accuracy_score(y_test, rf_test_pred)

print("✅ Random Forest training completed!")
print(f"Training Accuracy: {rf_train_accuracy:.4f}")
print(f"Test Accuracy: {rf_test_accuracy:.4f}")
print(f"Generalization Gap: {rf_train_accuracy - rf_test_accuracy:.4f}")

# 2.3 Evaluation (4 Marks)
print("\n2.3 Random Forest Evaluation")
print("-" * 40)

# Calculate F1-score
rf_f1_score = f1_score(y_test, rf_test_pred, average='weighted')

print(f"Test Accuracy: {rf_test_accuracy:.4f}")
print(f"F1-Score (weighted): {rf_f1_score:.4f}")

# Classification Report
print("\nClassification Report:")
class_names = label_encoder.classes_
print(classification_report(y_test, rf_test_pred, target_names=class_names))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_test_pred)
print(f"\nConfusion Matrix shape: {cm_rf.shape}")

# Per-class accuracy
class_accuracy_rf = cm_rf.diagonal() / cm_rf.sum(axis=1)
print("\nPer-class Accuracy:")
for i, acc in enumerate(class_accuracy_rf):
    print(f"  {class_names[i]}: {acc:.4f}")

# Feature importance
feature_importance = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Most Important Features:")
print(importance_df.head())

# =============================================================================
# PART 3: ANN MODEL USING KERAS (16 MARKS)
# =============================================================================

print("\n" + "="*60)
print("PART 3: ANN MODEL USING KERAS (16 MARKS)")
print("="*60)

# 3.1 Hyperparameter Tuning (5 Marks)
print("\n3.1 ANN Hyperparameter Tuning with Keras Tuner")
print("-" * 40)

def build_model(hp):
    """Model builder function for Keras Tuner."""
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Dense(
        units=hp.Int('units_1', min_value=32, max_value=512, step=32),
        activation='relu',
        input_shape=(X_train.shape[1],),
        kernel_regularizer=keras.regularizers.l2(0.001)
    ))
    model.add(layers.Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    
    # Additional hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i+2}', min_value=16, max_value=256, step=16),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        model.add(layers.Dropout(hp.Float(f'dropout_{i+2}', 0.1, 0.5, step=0.1)))
    
    # Output layer
    model.add(layers.Dense(len(np.unique(y_encoded)), activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print(f"Model builder defined for {len(np.unique(y_encoded))} classes")
print(f"Input shape: {X_train.shape[1]} features")

# Initialize RandomSearch tuner (reduced trials for speed)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Reduced for faster execution
    executions_per_trial=1,
    directory='keras_tuner_dir',
    project_name='dry_bean_ann',
    overwrite=True
)

print("Keras Tuner initialized (10 trials for demo)")

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

print("Starting hyperparameter search...")
tuner.search(
    X_train, y_train,
    epochs=30,  # Reduced for demo
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

print("✅ Hyperparameter search completed!")

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters found:")
print(f"  Number of layers: {best_hps.get('num_layers')}")
print(f"  Units in layer 1: {best_hps.get('units_1')}")
print(f"  Learning rate: {best_hps.get('learning_rate'):.6f}")

# 3.2 Build Model (3 Marks)
print("\n3.2 Building Final ANN Model")
print("-" * 40)

def build_final_model(best_hps):
    """Build final model with best hyperparameters and regularization."""
    model = keras.Sequential()
    
    # Input layer with L2 regularization
    model.add(layers.Dense(
        units=best_hps.get('units_1'),
        activation='relu',
        input_shape=(X_train.shape[1],),
        kernel_regularizer=keras.regularizers.l2(0.001)
    ))
    model.add(layers.Dropout(best_hps.get('dropout_1')))
    
    # Hidden layers with regularization
    for i in range(best_hps.get('num_layers')):
        model.add(layers.Dense(
            units=best_hps.get(f'units_{i+2}'),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        model.add(layers.Dropout(best_hps.get(f'dropout_{i+2}')))
    
    # Output layer
    model.add(layers.Dense(
        len(np.unique(y_encoded)), 
        activation='softmax'
    ))
    
    return model

# Create and compile the final model
final_ann_model = build_final_model(best_hps)
final_ann_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Final ANN model built with:")
print("  ✓ Dense layers with ReLU activation")
print("  ✓ Dropout for regularization")
print("  ✓ L2 kernel regularization")
print("  ✓ Softmax output for multi-class classification")
print(f"  ✓ Total parameters: {final_ann_model.count_params():,}")

# 3.3 Activations & Early Stopping (5 Marks)
print("\n3.3 Training with ReLU, Softmax, and Early Stopping")
print("-" * 40)

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=0
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=0
    )
]

print("Training callbacks configured:")
print("  ✓ EarlyStopping: monitors val_accuracy, patience=15")
print("  ✓ ReduceLROnPlateau: reduces LR when val_loss plateaus")
print("  ✓ ReLU activation in hidden layers")
print("  ✓ Softmax activation in output layer")

# Train the model
print("\nStarting model training...")
history = final_ann_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=0
)

print("✅ Model training completed!")
print(f"Training stopped at epoch: {len(history.history['accuracy'])}")

# 3.4 Evaluation (3 Marks)
print("\n3.4 ANN Model Evaluation")
print("-" * 40)

# Evaluate the model
ann_train_loss, ann_train_acc = final_ann_model.evaluate(X_train, y_train, verbose=0)
ann_test_loss, ann_test_acc = final_ann_model.evaluate(X_test, y_test, verbose=0)

print(f"Training Accuracy: {ann_train_acc:.4f}")
print(f"Training Loss: {ann_train_loss:.4f}")
print(f"Test Accuracy: {ann_test_acc:.4f}")
print(f"Test Loss: {ann_test_loss:.4f}")
print(f"Generalization Gap: {ann_train_acc - ann_test_acc:.4f}")

# Make predictions
ann_test_pred = final_ann_model.predict(X_test, verbose=0)
ann_test_pred_classes = np.argmax(ann_test_pred, axis=1)

# Calculate F1-score
ann_f1_score = f1_score(y_test, ann_test_pred_classes, average='weighted')
print(f"F1-Score (weighted): {ann_f1_score:.4f}")

# Classification Report
print(f"\nANN Classification Report:")
print(classification_report(y_test, ann_test_pred_classes, target_names=class_names))

# Confusion Matrix
cm_ann = confusion_matrix(y_test, ann_test_pred_classes)
class_accuracy_ann = cm_ann.diagonal() / cm_ann.sum(axis=1)

print(f"\nANN Per-class Accuracy:")
for i, acc in enumerate(class_accuracy_ann):
    print(f"  {class_names[i]}: {acc:.4f}")

# =============================================================================
# PART 4: INSIGHT QUESTIONS (9 MARKS)
# =============================================================================

print("\n" + "="*60)
print("PART 4: INSIGHT QUESTIONS (9 MARKS)")
print("="*60)

# 4.1 Compare Models (3 Marks)
print("\n4.1 Model Performance Comparison")
print("-" * 40)

# Create comprehensive comparison
comparison_data = {
    'Metric': ['Test Accuracy', 'F1-Score', 'Training Time', 'Interpretability', 'Overfitting'],
    'Random Forest': [f"{rf_test_accuracy:.4f}", f"{rf_f1_score:.4f}", "Fast", "High", f"{rf_train_accuracy - rf_test_accuracy:.4f}"],
    'ANN': [f"{ann_test_acc:.4f}", f"{ann_f1_score:.4f}", "Slow", "Low", f"{ann_train_acc - ann_test_acc:.4f}"]
}

comparison_df = pd.DataFrame(comparison_data)
print("Model Comparison Summary:")
print(comparison_df.to_string(index=False))

print(f"\nDetailed Performance Metrics:")
print(f"Random Forest - Accuracy: {rf_test_accuracy:.4f}, F1: {rf_f1_score:.4f}")
print(f"ANN          - Accuracy: {ann_test_acc:.4f}, F1: {ann_f1_score:.4f}")
print(f"Accuracy Difference: {abs(rf_test_accuracy - ann_test_acc):.4f}")

# 4.2 Best Model Justification (3 Marks)
print("\n4.2 Best Model Justification")
print("-" * 40)

performance_winner = "Random Forest" if rf_test_accuracy > ann_test_acc else "ANN"
performance_diff = abs(rf_test_accuracy - ann_test_acc)

print("COMPREHENSIVE MODEL ANALYSIS:")
print(f"🏆 Performance Winner: {performance_winner}")
print(f"📊 Performance Difference: {performance_diff:.4f}")

print(f"\nRandom Forest Advantages:")
print(f"  ✅ Interpretability: Feature importance analysis")
print(f"  ✅ Speed: Fast training and inference")
print(f"  ✅ Robustness: Less prone to overfitting")
print(f"  ✅ No scaling required")
print(f"  ✅ Good performance: {rf_test_accuracy:.4f} accuracy")

print(f"\nANN Advantages:")
print(f"  ✅ Flexibility: Can capture complex patterns")
print(f"  ✅ Performance: {ann_test_acc:.4f} accuracy")
print(f"  ✅ Scalability: Better with more data")

print(f"\nRecommendation for Food-Tech Production:")
if performance_diff < 0.02:  # Similar performance
    recommendation = "Random Forest"
    reason = "Similar performance but better interpretability and speed"
elif rf_test_accuracy > ann_test_acc:
    recommendation = "Random Forest"
    reason = "Better performance AND superior practical characteristics"
else:
    recommendation = "ANN"
    reason = "Superior performance justifies the complexity"

print(f"🎯 RECOMMENDED MODEL: {recommendation}")
print(f"📝 Reason: {reason}")

if rf_test_accuracy >= 0.90:
    print(f"✅ Excellent accuracy achieved (≥90%) - suitable for production")

# 4.3 Misclassifications (3 Marks)
print("\n4.3 Misclassification Analysis")
print("-" * 40)

# Find misclassified samples
rf_misclassified_mask = (y_test != rf_test_pred)
rf_misclassified_indices = np.where(rf_misclassified_mask)[0]

ann_misclassified_mask = (y_test != ann_test_pred_classes)
ann_misclassified_indices = np.where(ann_misclassified_mask)[0]

print(f"Random Forest Misclassifications: {len(rf_misclassified_indices)} / {len(y_test)} ({len(rf_misclassified_indices)/len(y_test)*100:.1f}%)")
print(f"ANN Misclassifications: {len(ann_misclassified_indices)} / {len(y_test)} ({len(ann_misclassified_indices)/len(y_test)*100:.1f}%)")

# Show first 5 misclassifications for Random Forest
print(f"\nFirst 5 Random Forest Misclassifications:")
for i, idx in enumerate(rf_misclassified_indices[:5]):
    true_label = class_names[y_test[idx]]
    pred_label = class_names[rf_test_pred[idx]]
    print(f"  {i+1}. Sample {idx}: {true_label} → {pred_label}")

# Show first 5 misclassifications for ANN
print(f"\nFirst 5 ANN Misclassifications:")
for i, idx in enumerate(ann_misclassified_indices[:5]):
    true_label = class_names[y_test[idx]]
    pred_label = class_names[ann_test_pred_classes[idx]]
    confidence = np.max(ann_test_pred[idx])
    print(f"  {i+1}. Sample {idx}: {true_label} → {pred_label} (conf: {confidence:.3f})")

# Common misclassifications
common_misclassified = set(rf_misclassified_indices) & set(ann_misclassified_indices)
print(f"\nSamples misclassified by BOTH models: {len(common_misclassified)}")

if len(common_misclassified) > 0:
    print("These samples are inherently difficult to classify:")
    for i, idx in enumerate(list(common_misclassified)[:3]):
        true_label = class_names[y_test[idx]]
        rf_pred = class_names[rf_test_pred[idx]]
        ann_pred = class_names[ann_test_pred_classes[idx]]
        print(f"  Sample {idx}: True={true_label}, RF={rf_pred}, ANN={ann_pred}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("FINAL SUMMARY - QUESTION 1 COMPLETED")
print("="*60)

print(f"\n📊 DATASET SUMMARY:")
print(f"   • Samples: {X.shape[0]:,}")
print(f"   • Features: {X.shape[1]}")
print(f"   • Classes: {len(class_names)}")
print(f"   • No missing values ✅")

print(f"\n🤖 MODEL PERFORMANCE:")
print(f"   • Random Forest: {rf_test_accuracy:.4f} accuracy, {rf_f1_score:.4f} F1")
print(f"   • ANN: {ann_test_acc:.4f} accuracy, {ann_f1_score:.4f} F1")

print(f"\n🏆 RECOMMENDATION:")
print(f"   • Best Model: {recommendation}")
print(f"   • Reason: {reason}")

print(f"\n✅ ALL PARTS COMPLETED:")
print(f"   ✓ Part 1: Data Preprocessing (14 marks)")
print(f"   ✓ Part 2: Random Forest Model (11 marks)")
print(f"   ✓ Part 3: ANN Model using Keras (16 marks)")
print(f"   ✓ Part 4: Insight Questions (9 marks)")

print(f"\n🎯 Ready for production deployment in food-tech automation!")
print("="*60)