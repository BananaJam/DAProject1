import streamlit as st
import pandas as pd
from typing import Optional

def download_csv_button(df: pd.DataFrame, label: str = "Download CSV", key: Optional[str] = None, filename: str = "data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )

    
def export_data():
    st.header("📥 Export Data")
    df = st.session_state.df
    
    if df.empty:
        st.warning("No data loaded yet. Go to 'Upload' to load a dataset.")
        return
    
    if not st.session_state.get("dataset_validated", False):
        st.warning("Please upload a valid dataset with all required columns first.")
        return
    
    # Check if predictions exist
    has_predictions = 'results_df' in st.session_state
    
    if has_predictions:
        st.subheader("📊 Export with Predictions")
        st.write("Download the dataset with churn predictions included.")
        
        results_df = st.session_state.results_df.copy()
        download_csv_button(
            results_df, 
            label="📥 Download Full Data with Predictions (CSV)", 
            key="dl_full_predictions",
            filename="churn_predictions.csv"
        )
        
        st.markdown("---")
        st.subheader("🔍 Filtered Export")
        st.write("Filter data before exporting.")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox(
                "Filter by",
                ["Predicted Churn", "Churn Probability", "Column Value"],
                key="filter_type"
            )
        
        if filter_type == "Predicted Churn":
            with col2:
                churn_status = st.selectbox(
                    "Churn Status",
                    ["All", "Predicted Churn", "No Churn"],
                    key="churn_filter"
                )
            
            if churn_status == "Predicted Churn":
                filtered = results_df[results_df['Predicted_Churn'] == 1]
            elif churn_status == "No Churn":
                filtered = results_df[results_df['Predicted_Churn'] == 0]
            else:
                filtered = results_df
            
            if len(filtered) > 0:
                st.dataframe(filtered.head(50), use_container_width=True)
                download_csv_button(
                    filtered, 
                    label=f"📥 Download Filtered Data ({len(filtered)} rows)", 
                    key="dl_filtered_churn",
                    filename=f"churn_filtered_{churn_status.lower().replace(' ', '_')}.csv"
                )
            else:
                st.info("No data matches the selected filter.")
        
        elif filter_type == "Churn Probability":
            with col2:
                prob_threshold = st.slider(
                    "Minimum Churn Probability",
                    0.0, 1.0, 0.5, 0.05,
                    key="prob_threshold"
                )
            
            filtered = results_df[results_df['Churn_Probability'] >= prob_threshold]
            
            if len(filtered) > 0:
                st.metric("High Risk Customers", len(filtered))
                st.dataframe(filtered.head(50), use_container_width=True)
                download_csv_button(
                    filtered, 
                    label=f"📥 Download High Risk Customers ({len(filtered)} rows)", 
                    key="dl_filtered_prob",
                    filename=f"high_risk_customers_threshold_{prob_threshold:.2f}.csv"
                )
            else:
                st.info(f"No customers found with churn probability >= {prob_threshold:.0%}")
        
        else:  # Column Value
            with col2:
                col = st.selectbox("Column", df.columns, key="filter_col")
                unique_vals = df[col].dropna().unique().tolist()
                val = st.selectbox("Value", unique_vals, key="filter_val")
            
            filtered = results_df[results_df[col] == val]
            
            if len(filtered) > 0:
                st.dataframe(filtered.head(50), use_container_width=True)
                download_csv_button(
                    filtered, 
                    label=f"📥 Download Filtered Data ({len(filtered)} rows)", 
                    key="dl_filtered_col",
                    filename=f"filtered_{col}_{val}.csv"
                )
            else:
                st.info("No data matches the selected filter.")
    
    else:
        st.subheader("📊 Export Original Data")
        st.write("Download the original dataset as CSV.")
        st.info("💡 Tip: Generate predictions in the 'Explore' section to export data with churn predictions.")
        
        download_csv_button(
            df, 
            label="📥 Download Original Data (CSV)", 
            key="dl_full",
            filename="customer_data.csv"
        )
        
        st.markdown("---")
        st.subheader("🔍 Quick Filter (Optional)")
        col = st.selectbox("Column", df.columns, key="filter_col_orig")
        unique_vals = df[col].dropna().unique().tolist()
        val = st.selectbox("Value", unique_vals, key="filter_val_orig")
        filtered = df[df[col] == val]
        
        if len(filtered) > 0:
            st.dataframe(filtered.head(50), use_container_width=True)
            download_csv_button(
                filtered, 
                label=f"📥 Download Filtered CSV ({len(filtered)} rows)", 
                key="dl_filtered",
                filename=f"filtered_{col}_{val}.csv"
            )
        else:
            st.info("No data matches the selected filter.")