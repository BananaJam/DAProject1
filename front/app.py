import streamlit as st
from upload import upload_file
from explore import explore_data
from visual import visualize_data
from about import about_page
from export import export_data
from state import ensure_df
from models import get_utils


# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------
# Initialize models on startup
# ---------------------------
@st.cache_resource
def init_models():
    """Initialize ML models on startup"""
    try:
        utils, models_loaded = get_utils()
        return utils, models_loaded
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, {}


# Initialize models
utils, models_loaded = init_models()


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("🎯 Churn Prediction App")

section = st.sidebar.radio(
    "Navigation",
    ["Upload", "Explore", "Visualize", "Export", "About"],
    index=0,
)

st.sidebar.markdown("---")

# Show model status
st.sidebar.subheader("Model Status")
if models_loaded:
    for model_name, loaded in models_loaded.items():
        status = "✅" if loaded else "❌"
        st.sidebar.write(f"{status} {model_name.replace('_', ' ').title()}")
else:
    st.sidebar.warning("Models not loaded")

st.sidebar.markdown("---")
st.sidebar.info("""
**Instructions:**
1. Upload a CSV/Excel file with customer data
2. Explore predictions in the Explore section
3. View visualizations in the Visualize section
4. Export results if needed
""")


# Keep a single source of truth for the working dataframe
ensure_df()


# ---------------------------
# Sections
# ---------------------------
st.title("📊 Customer Churn Prediction Analysis")
st.caption("Upload customer data to predict churn probability using machine learning models")


if section == "About":
    about_page()

elif section == "Upload":
    upload_file()

elif section == "Explore":
    explore_data()

elif section == "Visualize":
    visualize_data()


elif section == "Export":
    export_data()

