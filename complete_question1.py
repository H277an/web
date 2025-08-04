#!/usr/bin/env python3
"""
Complete Question 1 notebook with all remaining sections
"""

import json

def add_remaining_cells():
    """Add all remaining cells to Question 1 notebook"""
    
    # Load existing notebook
    with open("question1_dry_bean_classification.ipynb", 'r') as f:
        notebook = json.load(f)
    
    # Additional cells to add
    additional_cells = [
        # Class Distribution Analysis
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Class distribution analysis\n",
                "print(\"=== CLASS DISTRIBUTION ===\")\n",
                "class_counts = df['Class'].value_counts()\n",
                "class_percentages = df['Class'].value_counts(normalize=True) * 100\n",
                "\n",
                "class_distribution = pd.DataFrame({\n",
                "    'Count': class_counts,\n",
                "    'Percentage': class_percentages.round(2)\n",
                "})\n",
                "\n",
                "print(class_distribution)\n",
                "\n",
                "# Visualize class distribution\n",
                "plt.figure(figsize=(15, 5))\n",
                "\n",
                "plt.subplot(1, 2, 1)\n",
                "sns.countplot(data=df, x='Class', order=class_counts.index)\n",
                "plt.title('Bean Class Distribution (Count)')\n",
                "plt.xticks(rotation=45)\n",
                "plt.ylabel('Count')\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)\n",
                "plt.title('Bean Class Distribution (Percentage)')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(f\"\\nNumber of classes: {df['Class'].nunique()}\")\n",
                "print(f\"Classes: {list(df['Class'].unique())}\")"
            ]
        },
        
        # Label Encoding Section
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1.2 Label Encoding - 3 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Label Encoding for the target variable\n",
                "print(\"=== LABEL ENCODING ===\")\n",
                "\n",
                "# Create LabelEncoder\n",
                "label_encoder = LabelEncoder()\n",
                "\n",
                "# Original class labels\n",
                "print(\"Original classes:\")\n",
                "original_classes = df['Class'].unique()\n",
                "print(original_classes)\n",
                "\n",
                "# Encode the labels\n",
                "df['Class_Encoded'] = label_encoder.fit_transform(df['Class'])\n",
                "\n",
                "# Show the mapping\n",
                "print(\"\\nClass encoding mapping:\")\n",
                "class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))\n",
                "for original, encoded in class_mapping.items():\n",
                "    print(f\"{original} -> {encoded}\")\n",
                "\n",
                "# Verify encoding\n",
                "print(\"\\nEncoded classes:\")\n",
                "print(f\"Unique encoded values: {sorted(df['Class_Encoded'].unique())}\")\n",
                "print(f\"Number of classes: {df['Class_Encoded'].nunique()}\")"
            ]
        },
        
        # Feature Scaling Section
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1.3 Feature Scaling - 3 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature Scaling using StandardScaler\n",
                "print(\"=== FEATURE SCALING ===\")\n",
                "\n",
                "# Separate features and target\n",
                "numeric_features = df.select_dtypes(include=[np.number]).columns\n",
                "numeric_features = numeric_features.drop('Class_Encoded')  # Remove target from features\n",
                "X = df[numeric_features].copy()\n",
                "y = df['Class_Encoded'].copy()\n",
                "\n",
                "print(f\"Features shape: {X.shape}\")\n",
                "print(f\"Target shape: {y.shape}\")\n",
                "\n",
                "# Apply StandardScaler\n",
                "scaler = StandardScaler()\n",
                "X_scaled = scaler.fit_transform(X)\n",
                "X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)\n",
                "\n",
                "print(\"\\n=== SCALING VERIFICATION ===\")\n",
                "print(f\"Mean of scaled features (should be ~0): {X_scaled_df.mean().round(6).head()}\")\n",
                "print(f\"Std of scaled features (should be ~1): {X_scaled_df.std().round(6).head()}\")\n",
                "\n",
                "print(\"\\n✅ Feature scaling completed successfully!\")"
            ]
        },
        
        # Train-Test Split Section
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1.4 Train-Test Split with Stratification - 3 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stratified Train-Test Split\n",
                "print(\"=== STRATIFIED TRAIN-TEST SPLIT ===\")\n",
                "\n",
                "# Split the data with stratification to maintain class balance\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X_scaled, y, \n",
                "    test_size=0.2, \n",
                "    random_state=42, \n",
                "    stratify=y\n",
                ")\n",
                "\n",
                "print(f\"Training set shape: {X_train.shape}\")\n",
                "print(f\"Test set shape: {X_test.shape}\")\n",
                "print(f\"Training target shape: {y_train.shape}\")\n",
                "print(f\"Test target shape: {y_test.shape}\")\n",
                "\n",
                "# Verify stratification\n",
                "train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index() * 100\n",
                "test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index() * 100\n",
                "original_dist = y.value_counts(normalize=True).sort_index() * 100\n",
                "\n",
                "print(\"\\n=== STRATIFICATION VERIFICATION ===\")\n",
                "print(\"Class distribution comparison:\")\n",
                "for i in range(len(original_dist)):\n",
                "    print(f\"Class {i}: Original {original_dist.iloc[i]:.1f}%, Train {train_dist.iloc[i]:.1f}%, Test {test_dist.iloc[i]:.1f}%\")\n",
                "\n",
                "print(\"\\n✅ Data preprocessing completed successfully!\")"
            ]
        },
        
        # Random Forest Section
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "\n",
                "## Part 2: Random Forest Model (11 Marks) <a id=\"part2\"></a>\n",
                "\n",
                "### 2.1 Hyperparameter Tuning - 5 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Random Forest Hyperparameter Tuning\n",
                "print(\"=== RANDOM FOREST HYPERPARAMETER TUNING ===\")\n",
                "\n",
                "# Define parameter grid\n",
                "param_grid = {\n",
                "    'n_estimators': [50, 100, 200, 300],\n",
                "    'max_depth': [None, 10, 20, 30],\n",
                "    'min_samples_split': [2, 5, 10],\n",
                "    'min_samples_leaf': [1, 2, 4],\n",
                "    'bootstrap': [True, False],\n",
                "    'criterion': ['gini', 'entropy']\n",
                "}\n",
                "\n",
                "# Create base model\n",
                "rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)\n",
                "\n",
                "# Use RandomizedSearchCV for efficiency\n",
                "print(\"Starting hyperparameter search...\")\n",
                "rf_random_search = RandomizedSearchCV(\n",
                "    estimator=rf_base,\n",
                "    param_distributions=param_grid,\n",
                "    n_iter=50,\n",
                "    cv=5,\n",
                "    scoring='accuracy',\n",
                "    n_jobs=-1,\n",
                "    random_state=42,\n",
                "    verbose=1\n",
                ")\n",
                "\n",
                "# Fit the search\n",
                "rf_random_search.fit(X_train, y_train)\n",
                "\n",
                "print(f\"\\nBest cross-validation score: {rf_random_search.best_score_:.4f}\")\n",
                "print(f\"Best parameters: {rf_random_search.best_params_}\")\n",
                "\n",
                "# Store best model\n",
                "best_rf_model = rf_random_search.best_estimator_"
            ]
        },
        
        # Random Forest Evaluation
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.2 Model Training and Evaluation - 6 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Random Forest Model Evaluation\n",
                "print(\"=== RANDOM FOREST MODEL EVALUATION ===\")\n",
                "\n",
                "# Make predictions\n",
                "y_pred_rf = best_rf_model.predict(X_test)\n",
                "y_pred_proba_rf = best_rf_model.predict_proba(X_test)\n",
                "\n",
                "# Calculate metrics\n",
                "test_accuracy_rf = accuracy_score(y_test, y_pred_rf)\n",
                "f1_macro_rf = f1_score(y_test, y_pred_rf, average='macro')\n",
                "f1_weighted_rf = f1_score(y_test, y_pred_rf, average='weighted')\n",
                "\n",
                "print(f\"Test Accuracy: {test_accuracy_rf:.4f} ({test_accuracy_rf*100:.2f}%)\")\n",
                "print(f\"F1-Score (Macro): {f1_macro_rf:.4f}\")\n",
                "print(f\"F1-Score (Weighted): {f1_weighted_rf:.4f}\")\n",
                "\n",
                "# Classification report\n",
                "class_names = [label_encoder.inverse_transform([i])[0] for i in sorted(y.unique())]\n",
                "print(\"\\n=== CLASSIFICATION REPORT ===\")\n",
                "print(classification_report(y_test, y_pred_rf, target_names=class_names))\n",
                "\n",
                "# Confusion Matrix\n",
                "cm_rf = confusion_matrix(y_test, y_pred_rf)\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', \n",
                "            xticklabels=class_names, yticklabels=class_names)\n",
                "plt.title('Random Forest - Confusion Matrix')\n",
                "plt.xlabel('Predicted Class')\n",
                "plt.ylabel('True Class')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Feature importance\n",
                "feature_importance_rf = best_rf_model.feature_importances_\n",
                "importance_df = pd.DataFrame({\n",
                "    'Feature': X.columns,\n",
                "    'Importance': feature_importance_rf\n",
                "}).sort_values('Importance', ascending=False)\n",
                "\n",
                "plt.figure(figsize=(12, 8))\n",
                "sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')\n",
                "plt.title('Random Forest - Top 10 Feature Importances')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(\"\\n✅ Random Forest evaluation completed!\")"
            ]
        },
        
        # ANN Section
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "\n",
                "## Part 3: ANN Model using Keras (16 Marks) <a id=\"part3\"></a>\n",
                "\n",
                "### 3.1 Hyperparameter Tuning with Keras Tuner - 5 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ANN Hyperparameter Tuning\n",
                "print(\"=== ANN HYPERPARAMETER TUNING ===\")\n",
                "\n",
                "# Prepare data\n",
                "num_classes = len(np.unique(y))\n",
                "input_dim = X_train.shape[1]\n",
                "\n",
                "print(f\"Input dimension: {input_dim}\")\n",
                "print(f\"Number of classes: {num_classes}\")\n",
                "\n",
                "# Convert to numpy arrays\n",
                "X_train_nn = np.array(X_train)\n",
                "X_test_nn = np.array(X_test)\n",
                "y_train_nn = np.array(y_train)\n",
                "y_test_nn = np.array(y_test)\n",
                "\n",
                "# Model building function for Keras Tuner\n",
                "def build_model(hp):\n",
                "    model = keras.Sequential()\n",
                "    \n",
                "    # Input layer\n",
                "    model.add(layers.Dense(\n",
                "        units=hp.Int('input_units', min_value=32, max_value=256, step=32),\n",
                "        activation='relu',\n",
                "        input_shape=(input_dim,)\n",
                "    ))\n",
                "    model.add(layers.Dropout(hp.Float('input_dropout', 0.1, 0.5, step=0.1)))\n",
                "    \n",
                "    # Hidden layers\n",
                "    num_layers = hp.Int('num_layers', 1, 3)\n",
                "    for i in range(num_layers):\n",
                "        model.add(layers.Dense(\n",
                "            units=hp.Int(f'layer_{i}_units', min_value=32, max_value=128, step=32),\n",
                "            activation='relu'\n",
                "        ))\n",
                "        model.add(layers.Dropout(hp.Float(f'layer_{i}_dropout', 0.1, 0.5, step=0.1)))\n",
                "    \n",
                "    # Output layer\n",
                "    model.add(layers.Dense(num_classes, activation='softmax'))\n",
                "    \n",
                "    # Compile\n",
                "    model.compile(\n",
                "        optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),\n",
                "        loss='sparse_categorical_crossentropy',\n",
                "        metrics=['accuracy']\n",
                "    )\n",
                "    \n",
                "    return model\n",
                "\n",
                "# Create tuner\n",
                "tuner = RandomSearch(\n",
                "    build_model,\n",
                "    objective='val_accuracy',\n",
                "    max_trials=10,  # Reduced for demo\n",
                "    executions_per_trial=1,\n",
                "    directory='keras_tuner_dir',\n",
                "    project_name='dry_bean_classification'\n",
                ")\n",
                "\n",
                "# Early stopping\n",
                "early_stopping = EarlyStopping(\n",
                "    monitor='val_accuracy',\n",
                "    patience=5,\n",
                "    restore_best_weights=True,\n",
                "    verbose=0\n",
                ")\n",
                "\n",
                "# Search for best hyperparameters\n",
                "print(\"Starting hyperparameter search...\")\n",
                "tuner.search(\n",
                "    X_train_nn, y_train_nn,\n",
                "    epochs=20,\n",
                "    validation_split=0.2,\n",
                "    callbacks=[early_stopping],\n",
                "    verbose=0\n",
                ")\n",
                "\n",
                "# Get best hyperparameters\n",
                "best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]\n",
                "print(f\"\\nBest hyperparameters found:\")\n",
                "print(f\"Input units: {best_hps.get('input_units')}\")\n",
                "print(f\"Number of layers: {best_hps.get('num_layers')}\")\n",
                "print(f\"Optimizer: {best_hps.get('optimizer')}\")"
            ]
        },
        
        # ANN Training
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.2 Build and Train ANN Model - 8 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Build and train the best ANN model\n",
                "print(\"=== BUILDING AND TRAINING ANN MODEL ===\")\n",
                "\n",
                "# Build model with best hyperparameters\n",
                "best_model = tuner.hypermodel.build(best_hps)\n",
                "\n",
                "print(\"\\nModel architecture:\")\n",
                "best_model.summary()\n",
                "\n",
                "# Define callbacks\n",
                "early_stopping = EarlyStopping(\n",
                "    monitor='val_accuracy',\n",
                "    patience=10,\n",
                "    restore_best_weights=True,\n",
                "    verbose=1\n",
                ")\n",
                "\n",
                "# Train the model\n",
                "print(\"\\nTraining the model...\")\n",
                "history = best_model.fit(\n",
                "    X_train_nn, y_train_nn,\n",
                "    epochs=50,\n",
                "    batch_size=32,\n",
                "    validation_split=0.2,\n",
                "    callbacks=[early_stopping],\n",
                "    verbose=1\n",
                ")\n",
                "\n",
                "print(f\"\\nTraining completed after {len(history.history['loss'])} epochs\")"
            ]
        },
        
        # ANN Evaluation
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.3 ANN Model Evaluation - 3 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ANN Model Evaluation\n",
                "print(\"=== ANN MODEL EVALUATION ===\")\n",
                "\n",
                "# Make predictions\n",
                "y_pred_proba_ann = best_model.predict(X_test_nn, verbose=0)\n",
                "y_pred_ann = np.argmax(y_pred_proba_ann, axis=1)\n",
                "\n",
                "# Calculate metrics\n",
                "test_accuracy_ann = accuracy_score(y_test_nn, y_pred_ann)\n",
                "f1_macro_ann = f1_score(y_test_nn, y_pred_ann, average='macro')\n",
                "f1_weighted_ann = f1_score(y_test_nn, y_pred_ann, average='weighted')\n",
                "\n",
                "print(f\"Test Accuracy: {test_accuracy_ann:.4f} ({test_accuracy_ann*100:.2f}%)\")\n",
                "print(f\"F1-Score (Macro): {f1_macro_ann:.4f}\")\n",
                "print(f\"F1-Score (Weighted): {f1_weighted_ann:.4f}\")\n",
                "\n",
                "# Classification report\n",
                "print(\"\\n=== CLASSIFICATION REPORT ===\")\n",
                "print(classification_report(y_test_nn, y_pred_ann, target_names=class_names))\n",
                "\n",
                "# Confusion Matrix\n",
                "cm_ann = confusion_matrix(y_test_nn, y_pred_ann)\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Greens', \n",
                "            xticklabels=class_names, yticklabels=class_names)\n",
                "plt.title('ANN Model - Confusion Matrix')\n",
                "plt.xlabel('Predicted Class')\n",
                "plt.ylabel('True Class')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Training history\n",
                "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
                "\n",
                "# Accuracy plot\n",
                "axes[0].plot(history.history['accuracy'], label='Training Accuracy')\n",
                "axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')\n",
                "axes[0].set_title('Model Accuracy')\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].set_ylabel('Accuracy')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True)\n",
                "\n",
                "# Loss plot\n",
                "axes[1].plot(history.history['loss'], label='Training Loss')\n",
                "axes[1].plot(history.history['val_loss'], label='Validation Loss')\n",
                "axes[1].set_title('Model Loss')\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].set_ylabel('Loss')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(\"\\n✅ ANN evaluation completed!\")"
            ]
        },
        
        # Model Comparison
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "\n",
                "## Part 4: Insight Questions (9 Marks) <a id=\"part4\"></a>\n",
                "\n",
                "### 4.1 Model Comparison with Visualizations - 3 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Comprehensive Model Comparison\n",
                "print(\"=== MODEL COMPARISON ===\")\n",
                "\n",
                "# Performance comparison\n",
                "comparison_data = {\n",
                "    'Model': ['Random Forest', 'ANN'],\n",
                "    'Accuracy': [test_accuracy_rf, test_accuracy_ann],\n",
                "    'F1-Score (Macro)': [f1_macro_rf, f1_macro_ann],\n",
                "    'F1-Score (Weighted)': [f1_weighted_rf, f1_weighted_ann]\n",
                "}\n",
                "\n",
                "comparison_df = pd.DataFrame(comparison_data)\n",
                "print(\"\\nPerformance Comparison:\")\n",
                "print(comparison_df.round(4))\n",
                "\n",
                "# Visualization\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "# Accuracy comparison\n",
                "models = ['Random Forest', 'ANN']\n",
                "accuracies = [test_accuracy_rf, test_accuracy_ann]\n",
                "axes[0].bar(models, accuracies, color=['skyblue', 'lightcoral'])\n",
                "axes[0].set_title('Model Accuracy Comparison')\n",
                "axes[0].set_ylabel('Accuracy')\n",
                "axes[0].set_ylim(0.8, 1.0)\n",
                "for i, v in enumerate(accuracies):\n",
                "    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center')\n",
                "\n",
                "# F1-Score comparison\n",
                "f1_scores = [f1_macro_rf, f1_macro_ann]\n",
                "axes[1].bar(models, f1_scores, color=['lightgreen', 'orange'])\n",
                "axes[1].set_title('F1-Score (Macro) Comparison')\n",
                "axes[1].set_ylabel('F1-Score')\n",
                "for i, v in enumerate(f1_scores):\n",
                "    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center')\n",
                "\n",
                "# Per-class accuracy comparison\n",
                "rf_class_acc = cm_rf.diagonal() / cm_rf.sum(axis=1)\n",
                "ann_class_acc = cm_ann.diagonal() / cm_ann.sum(axis=1)\n",
                "\n",
                "x = np.arange(len(class_names))\n",
                "width = 0.35\n",
                "\n",
                "axes[2].bar(x - width/2, rf_class_acc, width, label='Random Forest')\n",
                "axes[2].bar(x + width/2, ann_class_acc, width, label='ANN')\n",
                "axes[2].set_title('Per-Class Accuracy Comparison')\n",
                "axes[2].set_ylabel('Accuracy')\n",
                "axes[2].set_xticks(x)\n",
                "axes[2].set_xticklabels(class_names, rotation=45)\n",
                "axes[2].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(\"\\n✅ Model comparison completed!\")"
            ]
        },
        
        # Best Model Justification
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.2 Best Model Justification - 3 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Best Model Justification\n",
                "print(\"=== BEST MODEL JUSTIFICATION ===\")\n",
                "\n",
                "# Determine best model\n",
                "if test_accuracy_rf > test_accuracy_ann:\n",
                "    best_model_name = \"Random Forest\"\n",
                "    best_accuracy = test_accuracy_rf\n",
                "    best_f1 = f1_macro_rf\n",
                "else:\n",
                "    best_model_name = \"ANN (Neural Network)\"\n",
                "    best_accuracy = test_accuracy_ann\n",
                "    best_f1 = f1_macro_ann\n",
                "\n",
                "print(f\"\\n🏆 BEST MODEL: {best_model_name}\")\n",
                "print(f\"Best Accuracy: {best_accuracy:.4f}\")\n",
                "print(f\"Best F1-Score: {best_f1:.4f}\")\n",
                "\n",
                "print(\"\\n=== JUSTIFICATION ===\")\n",
                "\n",
                "if best_model_name == \"Random Forest\":\n",
                "    print(\"\\n🌲 Random Forest is the better model because:\")\n",
                "    print(f\"• Higher accuracy: {test_accuracy_rf:.4f} vs {test_accuracy_ann:.4f}\")\n",
                "    print(f\"• Better F1-score: {f1_macro_rf:.4f} vs {f1_macro_ann:.4f}\")\n",
                "    print(\"• High interpretability through feature importance\")\n",
                "    print(\"• Faster training and inference\")\n",
                "    print(\"• More robust to overfitting\")\n",
                "    print(\"• Less hyperparameter sensitivity\")\n",
                "    print(\"• Better for production deployment\")\n",
                "else:\n",
                "    print(\"\\n🧠 ANN is the better model because:\")\n",
                "    print(f\"• Higher accuracy: {test_accuracy_ann:.4f} vs {test_accuracy_rf:.4f}\")\n",
                "    print(f\"• Better F1-score: {f1_macro_ann:.4f} vs {f1_macro_rf:.4f}\")\n",
                "    print(\"• Better at capturing complex patterns\")\n",
                "    print(\"• More flexible architecture\")\n",
                "    print(\"• Scalable to larger datasets\")\n",
                "    print(\"• Advanced regularization techniques\")\n",
                "\n",
                "print(\"\\n=== BUSINESS RECOMMENDATION ===\")\n",
                "print(f\"For the dry bean sorting application, I recommend {best_model_name}\")\n",
                "print(\"based on superior performance metrics and practical considerations.\")"
            ]
        },
        
        # Misclassification Analysis
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.3 Misclassification Analysis - 3 Marks"
            ]
        },
        
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Misclassification Analysis\n",
                "print(\"=== MISCLASSIFICATION ANALYSIS ===\")\n",
                "\n",
                "# Find misclassified samples\n",
                "rf_misclassified_mask = y_test != y_pred_rf\n",
                "ann_misclassified_mask = y_test_nn != y_pred_ann\n",
                "\n",
                "rf_misclassified_indices = np.where(rf_misclassified_mask)[0]\n",
                "ann_misclassified_indices = np.where(ann_misclassified_mask)[0]\n",
                "\n",
                "print(f\"Random Forest misclassifications: {len(rf_misclassified_indices)}\")\n",
                "print(f\"ANN misclassifications: {len(ann_misclassified_indices)}\")\n",
                "\n",
                "print(\"\\n=== FIRST 5 RANDOM FOREST MISCLASSIFICATIONS ===\")\n",
                "for i in range(min(5, len(rf_misclassified_indices))):\n",
                "    idx = rf_misclassified_indices[i]\n",
                "    true_label = y_test.iloc[idx]\n",
                "    pred_label = y_pred_rf[idx]\n",
                "    true_class = label_encoder.inverse_transform([true_label])[0]\n",
                "    pred_class = label_encoder.inverse_transform([pred_label])[0]\n",
                "    \n",
                "    print(f\"\\nSample {i+1}:\")\n",
                "    print(f\"  True Label: {true_class} (encoded: {true_label})\")\n",
                "    print(f\"  Predicted Label: {pred_class} (encoded: {pred_label})\")\n",
                "\n",
                "print(\"\\n=== FIRST 5 ANN MISCLASSIFICATIONS ===\")\n",
                "for i in range(min(5, len(ann_misclassified_indices))):\n",
                "    idx = ann_misclassified_indices[i]\n",
                "    true_label = y_test_nn[idx]\n",
                "    pred_label = y_pred_ann[idx]\n",
                "    true_class = label_encoder.inverse_transform([true_label])[0]\n",
                "    pred_class = label_encoder.inverse_transform([pred_label])[0]\n",
                "    confidence = y_pred_proba_ann[idx].max()\n",
                "    \n",
                "    print(f\"\\nSample {i+1}:\")\n",
                "    print(f\"  True Label: {true_class} (encoded: {true_label})\")\n",
                "    print(f\"  Predicted Label: {pred_class} (encoded: {pred_label})\")\n",
                "    print(f\"  Confidence: {confidence:.3f}\")\n",
                "\n",
                "print(\"\\n✅ Misclassification analysis completed!\")\n",
                "print(\"\\n🎉 COMPLETE ASSESSMENT FINISHED!\")\n",
                "print(\"\\nAll parts completed successfully:\")\n",
                "print(\"✅ Part 1: Data Preprocessing (14 marks)\")\n",
                "print(\"✅ Part 2: Random Forest Model (11 marks)\")\n",
                "print(\"✅ Part 3: ANN Model (16 marks)\")\n",
                "print(\"✅ Part 4: Insights and Analysis (9 marks)\")\n",
                "print(f\"\\n🏆 Recommended Model: {best_model_name}\")"
            ]
        }
    ]
    
    # Add cells to notebook
    notebook['cells'].extend(additional_cells)
    
    # Save updated notebook
    with open("question1_dry_bean_classification.ipynb", 'w') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Question 1 notebook completed with all sections!")

if __name__ == "__main__":
    add_remaining_cells()