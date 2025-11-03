import streamlit as st
import pandas as pd
from typing import Optional

def download_csv_button(df: pd.DataFrame, label: str = "Download CSV", key: Optional[str] = None):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name="data.csv",
        mime="text/csv",
        key=key,
    )

    
def export_data():
    st.header("Export data")
    df = st.session_state.df
    if df.empty:
        st.warning("No data loaded yet. Go to 'Upload' to load a dataset.")
    else:
        st.write("Download the current dataframe as CSV.")
        download_csv_button(df, label="Download full data as CSV", key="dl_full")

        st.markdown("---")
        st.subheader("Quick filter (optional)")
        # Simple, safe filter: choose a column and a value to keep
        col = st.selectbox("Column", df.columns)
        unique_vals = df[col].dropna().unique().tolist()
        val = st.selectbox("Value", unique_vals)
        filtered = df[df[col] == val]
        st.dataframe(filtered.head(50), use_container_width=True)
        download_csv_button(filtered, label="Download filtered CSV", key="dl_filtered")