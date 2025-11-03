"""
Main entry point for the Customer Churn Prediction Streamlit application.
"""
import streamlit.web.cli as stcli
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the path to the app file
    app_file = Path(__file__).parent / "front" / "app.py"
    
    # Run Streamlit app
    sys.argv = ["streamlit", "run", str(app_file)]
    sys.exit(stcli.main())

