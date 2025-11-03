import streamlit as st
import pandas as pd
from typing import Optional
import plotly.express as px
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


def get_sample_data(choice: str) -> pd.DataFrame:
    # plotly express ships small sample datasets (no extra deps)
    if choice == "Iris":
        return px.data.iris()
    if choice == "Tips":
        return px.data.tips()
    if choice == "Gapminder":
        return px.data.gapminder()
    return pd.DataFrame()

def upload_file(sample_choice: Optional[str] = None, use_sample: bool = False):
    st.header("Upload a dataset")
    st.write("Upload a CSV or Excel file, or load a sample dataset from the sidebar.")

    uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

    col1, col2 = st.columns([1, 1])
    with col1:
        if uploaded is not None:
            df = load_uploaded_file(uploaded)
            if df is not None and not df.empty:
                set_df(df)
                st.success(f"Loaded {uploaded.name} with shape {df.shape}")
        else:
            st.info("No file selected yet.")

    with col2:
        if use_sample and sample_choice and sample_choice != "None":
            df = get_sample_data(sample_choice)
            set_df(df)
            st.success(f"Loaded sample '{sample_choice}' with shape {df.shape}")

    if "df" in st.session_state and not st.session_state.df.empty:
        st.subheader("Preview")
        n_rows = st.slider("Rows to show", 5, 100, 10)
        st.dataframe(st.session_state.df.head(n_rows), use_container_width=True)