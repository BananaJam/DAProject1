"""
Script to train and save all machine learning models for churn prediction.

This script:
1. Downloads the churn modelling dataset from Kaggle
2. Preprocesses the data using ClassifierUtils
3. Trains Logistic Regression, Random Forest, and XGBoost models
4. Evaluates all models
5. Saves all trained models
"""

import kagglehub
import pandas as pd
from ml_utils import ClassifierUtils

def main():
    print("=" * 60)
    print("CHURN PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Download dataset
    print("\n[1/5] Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("shrutimechlearn/churn-modelling")
    print(f"Dataset downloaded to: {path}")
    
    # Step 2: Load dataset
    print("\n[2/5] Loading dataset...")
    df = pd.read_csv(f"{path}/Churn_Modelling.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Step 3: Initialize ClassifierUtils
    print("\n[3/5] Initializing ClassifierUtils...")
    utils = ClassifierUtils(model_dir="models")
    
    # Step 4: Preprocess data
    print("\n[4/5] Preprocessing data...")
    X_train, y_train, X_test, y_test = utils.preprocess_data(
        df=df,
        target_column="Exited",
        drop_columns=['RowNumber', 'CustomerId', 'Surname'],
        categorical_columns=['Geography', 'Gender'],
        test_size=0.2,
        random_state=42,
        fit_preprocessing=True
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set churn rate: {y_train.mean():.2%}")
    print(f"Test set churn rate: {y_test.mean():.2%}")
    
    # Step 5: Train all models
    print("\n[5/5] Training all models...")
    
    # Calculate scale_pos_weight for XGBoost (to handle class imbalance)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Train all models with appropriate parameters
    models = utils.train_all_models(
        X_train=X_train,
        y_train=y_train,
        lr_params={
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        },
        rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        },
        xgb_params={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
    )
    
    print(f"✓ Trained {len(models)} models:")
    for model_name in models.keys():
        print(f"  - {model_name}")
    
    # Step 6: Evaluate all models
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    evaluation_results = utils.evaluate_all_models(X_test, y_test)
    print("\nEvaluation Results:")
    print(evaluation_results.to_string(index=False))
    
    # Step 7: Save all models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    saved_paths = utils.save_all_models()
    
    print("\n✓ All models saved successfully:")
    for model_name, filepath in saved_paths.items():
        print(f"  - {model_name}: {filepath}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nAll models saved in: {utils.model_dir}")
    print("\nYou can now load these models using:")
    print("  utils = ClassifierUtils()")
    print("  utils.load_model('models/logistic_regression.pkl')")
    print("  utils.load_model('models/random_forest.pkl')")
    print("  utils.load_model('models/xgboost.pkl')")

if __name__ == "__main__":
    main()
