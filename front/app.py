import io
from typing import Optional, Tuple
import pandas as pd
import plotly.express as px
import streamlit as st
from upload import upload_file
from explore import explore_data
from visual import visualize_data
from about import about_page
from export import export_data
from state import ensure_df


# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Data Analysis Template",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Controls")

section = st.sidebar.radio(
    "Go to",
    ["Upload", "Explore", "Visualize", "Export", "About"],
    index=0,
)

st.sidebar.markdown("---")

with st.sidebar.expander("Sample datasets", expanded=False):
    sample_choice = st.selectbox("Pick a sample", ["None", "Iris", "Tips", "Gapminder"], index=0)
    use_sample = st.button("Load sample", disabled=sample_choice == "None")



# Keep a single source of truth for the working dataframe
ensure_df()


# ---------------------------
# Sections
# ---------------------------
st.title("📊 Data Analysis UI Template")
st.caption("Streamlit starter UI for quick data exploration and visualization")


if section == "Upload":
    upload_file(sample_choice=sample_choice, use_sample=use_sample)

elif section == "Explore":
    explore_data()

elif section == "Visualize":
    visualize_data()

elif section == "Export":
    export_data()

elif section == "About":
    about_page()
