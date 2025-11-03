import streamlit as st
import pandas as pd
from typing import Tuple
from models import get_utils, predict_churn


def numeric_and_categorical(df: pd.DataFrame) -> Tuple[list, list]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def explore_data():
    st.header("🔍 Explore Data & Predictions")
    df = st.session_state.df
    
    if df.empty:
        st.warning("No data loaded yet. Go to 'Upload' to load a dataset.")
        return
    
    if not st.session_state.get("dataset_validated", False):
        st.warning("Please upload a valid dataset with all required columns first.")
        return
    
    # Model selection
    utils, models_loaded = get_utils()
    available_models = [name for name, loaded in models_loaded.items() if loaded]
    
    if not available_models:
        st.error("No ML models are loaded. Please ensure model files exist in the models/ directory.")
        return
    
    selected_model = st.selectbox(
        "Select Model for Predictions",
        available_models,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # KPIs
    num_cols, cat_cols = numeric_and_categorical(df)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Customers", len(df))
    with kpi2:
        st.metric("Columns", df.shape[1])
    with kpi3:
        st.metric("Numeric Features", len(num_cols))
    with kpi4:
        st.metric("Categorical Features", len(cat_cols))
    
    # Make predictions
    try:
        with st.spinner("Generating churn predictions..."):
            predictions, probabilities = predict_churn(df, model_name=selected_model)
            
            # Create results dataframe
            results_df = df.copy()
            results_df['Predicted_Churn'] = predictions
            results_df['Churn_Probability'] = probabilities
            results_df['Churn_Probability'] = results_df['Churn_Probability'].round(4)
            
            # Store results in session state
            st.session_state.predictions = predictions
            st.session_state.churn_probabilities = probabilities
            st.session_state.results_df = results_df
            
        # Prediction summary
        st.markdown("---")
        st.subheader("📊 Churn Prediction Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", total_customers)
        with col2:
            predicted_churns = int(predictions.sum())
            st.metric("Predicted Churns", predicted_churns, delta=f"{(predicted_churns/total_customers)*100:.1f}%")
        with col3:
            avg_probability = probabilities.mean()
            st.metric("Avg Churn Probability", f"{avg_probability:.2%}")
        with col4:
            high_risk = (probabilities >= 0.7).sum()
            st.metric("High Risk (≥70%)", high_risk, delta=f"{(high_risk/total_customers)*100:.1f}%")
        
        # Data table with predictions
        st.markdown("---")
        st.subheader("📋 Customer Data with Churn Predictions")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_only_churns = st.checkbox("Show only predicted churns", value=False)
        with col2:
            sort_by_prob = st.checkbox("Sort by churn probability (descending)", value=True)
        
        display_df = results_df.copy()
        
        if show_only_churns:
            display_df = display_df[display_df['Predicted_Churn'] == 1]
        
        if sort_by_prob:
            display_df = display_df.sort_values('Churn_Probability', ascending=False)
        
        # Format columns for better display
        display_df = display_df.copy()
        display_df['Churn_Probability'] = display_df['Churn_Probability'].apply(lambda x: f"{x:.2%}")
        
        # Show dataframe
        n_rows = st.slider("Rows to display", 10, min(500, len(display_df)), 50)
        st.dataframe(display_df.head(n_rows), use_container_width=True)
        
        # Download results
        st.download_button(
            label="📥 Download Results with Predictions (CSV)",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
        
        # Schema information
        st.markdown("---")
        with st.expander("📋 Dataset Schema Information"):
            st.dataframe(pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Non-Null": df.notna().sum().values,
                "Nulls": df.isna().sum().values,
            }), use_container_width=True)
        
        # Summary statistics
        with st.expander("📈 Summary Statistics (Numeric Features)"):
            if num_cols:
                st.dataframe(df[num_cols].describe().T, use_container_width=True)
            else:
                st.info("No numeric columns detected.")
    
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        st.info("Please ensure your dataset has all required columns and matches the expected format.")