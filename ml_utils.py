"""
Utility Library for Dataset Analysis with Pretrained Classifiers

This module provides tools for training, saving, loading, and using pretrained
machine learning models (Logistic Regression, Random Forest, XGBoost) for 
classification tasks. Designed for use in Streamlit applications.

Features:
- Data preprocessing (encoding, scaling)
- Model training and evaluation
- Model persistence (save/load)
- Prediction and probability estimation
- Comprehensive evaluation metrics
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple, Optional
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class ClassifierUtils:
    """
    Utility class for managing classification models and data preprocessing.
    
    Supports:
    - Logistic Regression
    - Random Forest
    - XGBoost
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the utility class.
        
        Args:
            model_dir: Directory to save/load models (default: "models")
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.scaler = None
        self.label_encoders = {}
        self.models = {}
        self.feature_names = None
        self.target_name = None
        
    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        drop_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        fit_preprocessing: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Preprocess data for machine learning.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            drop_columns: Columns to drop (e.g., IDs, names)
            categorical_columns: Columns to encode (if None, auto-detect)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            fit_preprocessing: Whether to fit scalers/encoders (True) or use existing (False)
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        df_processed = df.copy()
        
        # Drop specified columns
        if drop_columns:
            df_processed = df_processed.drop(columns=drop_columns, errors='ignore')
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        self.target_name = target_column
        
        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Encode categorical variables
        X_encoded = X.copy()
        for col in categorical_columns:
            if col in X.columns:
                if fit_preprocessing:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        X_encoded[col] = X[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        raise ValueError(f"Label encoder for {col} not found. Train preprocessing first.")
        
        # Store feature names
        if fit_preprocessing:
            self.feature_names = X_encoded.columns.tolist()
        
        # Ensure all features are numeric
        X_encoded = X_encoded.select_dtypes(include=[np.number])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        if fit_preprocessing:
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            if self.scaler is None:
                raise ValueError("Scaler not found. Train preprocessing first.")
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> LogisticRegression:
        """
        Train a Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional arguments for LogisticRegression
            
        Returns:
            Trained LogisticRegression model
        """
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }
        default_params.update(kwargs)
        
        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train a Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional arguments for RandomForestClassifier
            
        Returns:
            Trained RandomForestClassifier model
        """
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> xgb.XGBClassifier:
        """
        Train an XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional arguments for XGBClassifier
            
        Returns:
            Trained XGBClassifier model
        """
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 6,
            'learning_rate': 0.1,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        default_params.update(kwargs)
        
        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        lr_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None
    ) -> Dict[str, Union[LogisticRegression, RandomForestClassifier, xgb.XGBClassifier]]:
        """
        Train all three models.
        
        Args:
            X_train: Training features
            y_train: Training target
            lr_params: Parameters for Logistic Regression
            rf_params: Parameters for Random Forest
            xgb_params: Parameters for XGBoost
            
        Returns:
            Dictionary of trained models
        """
        if lr_params is None:
            lr_params = {}
        if rf_params is None:
            rf_params = {}
        if xgb_params is None:
            xgb_params = {}
        
        self.train_logistic_regression(X_train, y_train, **lr_params)
        self.train_random_forest(X_train, y_train, **rf_params)
        self.train_xgboost(X_train, y_train, **xgb_params)
        
        return self.models
    
    def predict(
        self,
        model_name: str,
        X: pd.DataFrame,
        use_scaler: bool = True
    ) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model ('logistic_regression', 'random_forest', 'xgboost')
            X: Features to predict on
            use_scaler: Whether to scale features before prediction
            
        Returns:
            Array of predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train or load model first.")
        
        X_processed = X.copy()
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Select only numeric columns
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        # Scale if needed
        if use_scaler and self.scaler is not None:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        model = self.models[model_name]
        return model.predict(X_processed)
    
    def predict_proba(
        self,
        model_name: str,
        X: pd.DataFrame,
        use_scaler: bool = True
    ) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of the model
            X: Features to predict on
            use_scaler: Whether to scale features before prediction
            
        Returns:
            Array of prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train or load model first.")
        
        X_processed = X.copy()
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Select only numeric columns
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        # Scale if needed
        if use_scaler and self.scaler is not None:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        model = self.models[model_name]
        return model.predict_proba(X_processed)
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_scaler: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a model and return comprehensive metrics.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            use_scaler: Whether to scale features before prediction
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(model_name, X_test, use_scaler=use_scaler)
        y_proba = self.predict_proba(model_name, X_test, use_scaler=use_scaler)
        
        # For binary classification, get probabilities of positive class
        if y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC (only for binary classification)
        try:
            if len(np.unique(y_test)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba_pos)
        except:
            metrics['roc_auc'] = None
        
        return metrics
    
    def evaluate_all_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate all trained models and return results as a DataFrame.
        
        Args:
            X_test: Test features
            y_test: Test target
            use_scaler: Whether to scale features before prediction
            
        Returns:
            DataFrame with evaluation metrics for all models
        """
        results = []
        for model_name in self.models.keys():
            metrics = self.evaluate_model(model_name, X_test, y_test, use_scaler=use_scaler)
            metrics['model'] = model_name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def get_feature_importance(
        self,
        model_name: str,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model ('random_forest' or 'xgboost')
            top_n: Number of top features to return (None for all)
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        if model_name == 'random_forest':
            importances = model.feature_importances_
        elif model_name == 'xgboost':
            importances = model.feature_importances_
        else:
            raise ValueError(f"Feature importance not available for {model_name}")
        
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def save_model(
        self,
        model_name: str,
        filepath: Optional[str] = None
    ) -> str:
        """
        Save a trained model and associated preprocessing objects.
        
        Args:
            model_name: Name of the model to save
            filepath: Optional custom filepath (default: model_dir/model_name.pkl)
            
        Returns:
            Path to saved file
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        if filepath is None:
            filepath = self.model_dir / f"{model_name}.pkl"
        else:
            filepath = Path(filepath)
        
        # Save model and preprocessing objects
        save_dict = {
            'model': self.models[model_name],
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        return str(filepath)
    
    def save_all_models(self) -> Dict[str, str]:
        """
        Save all trained models.
        
        Returns:
            Dictionary mapping model names to saved filepaths
        """
        saved_paths = {}
        for model_name in self.models.keys():
            saved_paths[model_name] = self.save_model(model_name)
        return saved_paths
    
    def load_model(
        self,
        filepath: str,
        model_name: Optional[str] = None
    ) -> str:
        """
        Load a pretrained model and associated preprocessing objects.
        
        Args:
            filepath: Path to the saved model file
            model_name: Optional name for the model (default: inferred from filename)
            
        Returns:
            Name of the loaded model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Infer model name from filename if not provided
        if model_name is None:
            model_name = filepath.stem
        
        # Restore objects
        self.models[model_name] = save_dict['model']
        self.scaler = save_dict.get('scaler', self.scaler)
        self.label_encoders = save_dict.get('label_encoders', self.label_encoders)
        self.feature_names = save_dict.get('feature_names', self.feature_names)
        self.target_name = save_dict.get('target_name', self.target_name)
        
        return model_name
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        info = {
            'model_type': type(model).__name__,
            'model_name': model_name,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'has_scaler': self.scaler is not None,
            'categorical_columns': list(self.label_encoders.keys())
        }
        
        # Add model-specific parameters
        if hasattr(model, 'get_params'):
            info['parameters'] = model.get_params()
        
        return info

