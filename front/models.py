"""
Module for managing ML models for churn prediction.
"""
import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ml_utils import ClassifierUtils


@st.cache_resource
def load_models():
    """
    Load all pretrained churn prediction models.
    This is cached to avoid reloading on every rerun.
    """
    models_dir = Path(__file__).parent.parent / "models"
    utils = ClassifierUtils(model_dir=str(models_dir))
    
    models_loaded = {}
    model_files = {
        'logistic_regression': 'logistic_regression.pkl',
        'random_forest': 'random_forest.pkl',
        'xgboost': 'xgboost.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                utils.load_model(str(model_path), model_name=model_name)
                models_loaded[model_name] = True
            except Exception as e:
                st.error(f"Failed to load {model_name}: {e}")
                models_loaded[model_name] = False
        else:
            st.warning(f"Model file not found: {model_path}")
            models_loaded[model_name] = False
    
    return utils, models_loaded


def get_utils():
    """
    Get the ClassifierUtils instance with loaded models.
    """
    if 'ml_utils' not in st.session_state:
        utils, models_loaded = load_models()
        st.session_state.ml_utils = utils
        st.session_state.models_loaded = models_loaded
    return st.session_state.ml_utils, st.session_state.models_loaded


def predict_churn(df, model_name='random_forest'):
    """
    Predict churn for a dataframe.
    
    Args:
        df: DataFrame with customer data
        model_name: Name of the model to use ('logistic_regression', 'random_forest', 'xgboost')
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    import pandas as pd
    
    utils, models_loaded = get_utils()
    
    if not models_loaded.get(model_name, False):
        raise ValueError(f"Model {model_name} is not loaded")
    
    # Prepare dataframe for prediction by dropping columns that were excluded during training
    df_for_prediction = df.copy()
    
    # Drop columns that were dropped during training (same as in training.py)
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    for col in columns_to_drop:
        if col in df_for_prediction.columns:
            df_for_prediction = df_for_prediction.drop(columns=[col])
    
    # Drop target column if it exists (Exited)
    if 'Exited' in df_for_prediction.columns:
        df_for_prediction = df_for_prediction.drop(columns=['Exited'])
    
    # Ensure we only use the feature columns that were used during training
    if utils.feature_names is not None:
        # Check if all required features are present
        missing_features = [f for f in utils.feature_names if f not in df_for_prediction.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")
        
        # Select only the feature columns in the correct order
        df_for_prediction = df_for_prediction[utils.feature_names]
    
    # Make predictions
    predictions = utils.predict(model_name, df_for_prediction, use_scaler=True)
    probabilities = utils.predict_proba(model_name, df_for_prediction, use_scaler=True)
    
    # Get probability of churn (positive class)
    churn_proba = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
    
    return predictions, churn_proba

