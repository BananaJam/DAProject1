import streamlit as st
import pandas as pd
from typing import Tuple


def numeric_and_categorical(df: pd.DataFrame) -> Tuple[list, list]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def explore_data():
    st.header("Explore data")
    df = st.session_state.df
    if df.empty:
        st.warning("No data loaded yet. Go to 'Upload' to load a dataset.")
    else:
        num_cols, cat_cols = numeric_and_categorical(df)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Rows", len(df))
        with kpi2:
            st.metric("Columns", df.shape[1])
        with kpi3:
            st.metric("Numeric", len(num_cols))
        with kpi4:
            st.metric("Categorical", len(cat_cols))

        st.markdown("---")
        st.subheader("Schema")
        st.dataframe(pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null": df.notna().sum().values,
            "nulls": df.isna().sum().values,
        }), use_container_width=True)

        st.subheader("Summary statistics (numeric)")
        if num_cols:
            st.dataframe(df[num_cols].describe().T, use_container_width=True)
        else:
            st.info("No numeric columns detected.")