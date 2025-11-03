import streamlit as st

def about_page():
    st.title("About This App")
    st.markdown("""
    This Data Analysis Template app is built using Streamlit and provides a simple interface for uploading, exploring, visualizing, and exporting datasets.

    **Features:**
    - Upload CSV or Excel files
    - Explore dataset schema and summary statistics
    - Visualize data with interactive plots
    - Export data to CSV format

    **Technologies Used:**
    - Streamlit for the web interface
    - Pandas for data manipulation
    - Plotly for data visualization

    **Usage:**
    1. Navigate to the 'Upload' page to upload your dataset.
    2. Go to the 'Explore' page to view dataset details.
    3. Use the 'Visualize' page to create plots.
    4. Export your data from the 'Export' page.

    Feel free to contribute or suggest improvements!
    """)