import pandas as pd
import streamlit as st


def ensure_df() -> None:
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()


def set_df(new_df: pd.DataFrame) -> None:
    st.session_state.df = new_df.copy()
