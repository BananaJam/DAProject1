import streamlit as st
import pandas as pd
from typing import Optional, Tuple
from state import set_df

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def _read_csv(upload) -> pd.DataFrame:
    return pd.read_csv(upload)


@st.cache_data(show_spinner=False)
def _read_excel(upload) -> pd.DataFrame:
    return pd.read_excel(upload)


def load_uploaded_file(upload) -> Optional[pd.DataFrame]:
    if upload is None:
        return None
    name = upload.name.lower()
    try:
        if name.endswith(".csv"):
            return _read_csv(upload)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return _read_excel(upload)
        st.warning("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


def validate_churn_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the dataset has the required columns for churn prediction.
    Expected columns: CreditScore, Geography, Gender, Age, Tenure, Balance,
                     NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    """
    required_cols = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    return True, "Dataset is valid for churn prediction"


def upload_file(sample_choice: Optional[str] = None, use_sample: bool = False):
    st.header("📤 Upload Customer Data")
    st.write("Upload a CSV or Excel file with customer data for churn prediction analysis.")
    
    st.info("""
    **Required columns for churn prediction:**
    - CreditScore, Geography, Gender, Age, Tenure
    - Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    """)

    uploaded = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=["csv", "xlsx", "xls"], 
        accept_multiple_files=False
    )

    if uploaded is not None:
        df = load_uploaded_file(uploaded)
        if df is not None and not df.empty:
            # Validate dataset
            is_valid, message = validate_churn_dataset(df)
            
            if is_valid:
                set_df(df)
                st.success(f"✓ Loaded {uploaded.name} with shape {df.shape}")
                st.session_state.dataset_validated = True
            else:
                st.error(f"Dataset validation failed: {message}")
                st.info("Please ensure your dataset contains all required columns.")
                st.session_state.dataset_validated = False
        else:
            st.session_state.dataset_validated = False
    else:
        st.info("No file selected yet.")
        st.session_state.dataset_validated = False

    if "df" in st.session_state and not st.session_state.df.empty and st.session_state.get("dataset_validated", False):
        st.markdown("---")
        st.subheader("📋 Data Preview")
        n_rows = st.slider("Rows to show", 5, 100, 10)
        st.dataframe(st.session_state.df.head(n_rows), use_container_width=True)
        
        # Show dataset info
        st.markdown("---")
        st.subheader("📊 Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(st.session_state.df))
        with col2:
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col3:
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
            st.metric("Numeric Columns", len(numeric_cols))