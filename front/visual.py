import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from explore import numeric_and_categorical
from models import get_utils


def visualize_data():
    st.header("📊 Churn Prediction Visualizations")
    df = st.session_state.df
    
    if df.empty:
        st.warning("No data loaded yet. Go to 'Upload' to load a dataset.")
        return
    
    if not st.session_state.get("dataset_validated", False):
        st.warning("Please upload a valid dataset with all required columns first.")
        return
    
    # Check if predictions exist
    if 'predictions' not in st.session_state or 'churn_probabilities' not in st.session_state:
        st.info("Please go to the 'Explore' section first to generate predictions.")
        return
    
    predictions = st.session_state.predictions
    probabilities = st.session_state.churn_probabilities
    results_df = st.session_state.results_df.copy()
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization",
        [
            "Churn Distribution",
            "Churn by Geography",
            "Churn by Gender",
            "Age Distribution by Churn",
            "Churn Probability Distribution",
            "Feature Correlation Analysis",
            "Feature Importance",
            "High Risk Customers Analysis"
        ]
    )
    
    st.markdown("---")
    
    if viz_type == "Churn Distribution":
        st.subheader("📊 Churn Distribution")
        churn_counts = pd.Series(predictions).value_counts().sort_index()
        fig = px.bar(
            x=['Not Churned', 'Churned'],
            y=churn_counts.values,
            color=['Not Churned', 'Churned'],
            color_discrete_map={'Not Churned': 'skyblue', 'Churned': 'salmon'},
            labels={'x': 'Churn Status', 'y': 'Count'},
            title='Distribution of Predicted Churns'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Not Churned", int(churn_counts.get(0, 0)))
        with col2:
            churn_rate = (churn_counts.get(1, 0) / len(predictions)) * 100
            st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    elif viz_type == "Churn by Geography":
        st.subheader("🌍 Churn by Geography")
        if 'Geography' in df.columns:
            churn_geo = pd.crosstab(df['Geography'], predictions, normalize='index') * 100
            churn_geo.columns = ['Not Churned', 'Churned']
            
            fig = px.bar(
                churn_geo.reset_index(),
                x='Geography',
                y=['Not Churned', 'Churned'],
                barmode='group',
                title='Churn Rate by Geography',
                labels={'value': 'Percentage', 'Geography': 'Geography'},
                color_discrete_map={'Not Churned': 'skyblue', 'Churned': 'salmon'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show actual counts
            count_geo = pd.crosstab(df['Geography'], predictions)
            count_geo.columns = ['Not Churned', 'Churned']
            st.dataframe(count_geo, use_container_width=True)
        else:
            st.warning("Geography column not found in dataset.")
    
    elif viz_type == "Churn by Gender":
        st.subheader("👥 Churn by Gender")
        if 'Gender' in df.columns:
            churn_gender = pd.crosstab(df['Gender'], predictions, normalize='index') * 100
            churn_gender.columns = ['Not Churned', 'Churned']
            
            fig = px.bar(
                churn_gender.reset_index(),
                x='Gender',
                y=['Not Churned', 'Churned'],
                barmode='group',
                title='Churn Rate by Gender',
                labels={'value': 'Percentage', 'Gender': 'Gender'},
                color_discrete_map={'Not Churned': 'skyblue', 'Churned': 'salmon'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show actual counts
            count_gender = pd.crosstab(df['Gender'], predictions)
            count_gender.columns = ['Not Churned', 'Churned']
            st.dataframe(count_gender, use_container_width=True)
        else:
            st.warning("Gender column not found in dataset.")
    
    elif viz_type == "Age Distribution by Churn":
        st.subheader("📈 Age Distribution by Churn Status")
        if 'Age' in df.columns:
            not_churned_ages = df[predictions == 0]['Age']
            churned_ages = df[predictions == 1]['Age']
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=not_churned_ages,
                name='Not Churned',
                opacity=0.7,
                marker_color='skyblue',
                nbinsx=30
            ))
            fig.add_trace(go.Histogram(
                x=churned_ages,
                name='Churned',
                opacity=0.7,
                marker_color='salmon',
                nbinsx=30
            ))
            
            fig.update_layout(
                title='Age Distribution by Churn Status',
                xaxis_title='Age',
                yaxis_title='Frequency',
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Age (Not Churned)", f"{not_churned_ages.mean():.1f}")
            with col2:
                st.metric("Avg Age (Churned)", f"{churned_ages.mean():.1f}")
        else:
            st.warning("Age column not found in dataset.")
    
    elif viz_type == "Churn Probability Distribution":
        st.subheader("📊 Churn Probability Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probabilities,
            nbinsx=50,
            name='Churn Probability',
            marker_color='lightcoral'
        ))
        fig.update_layout(
            title='Distribution of Churn Probabilities',
            xaxis_title='Churn Probability',
            yaxis_title='Frequency',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk categories
        risk_categories = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low (0-30%)', 'Medium (30-50%)', 'High (50-70%)', 'Very High (70-100%)']
        )
        risk_counts = risk_categories.value_counts().sort_index()
        
        fig2 = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title='Customers by Risk Category',
            labels={'x': 'Risk Category', 'y': 'Count'},
            color=risk_counts.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.dataframe(risk_counts.to_frame('Count'), use_container_width=True)
    
    elif viz_type == "Feature Correlation Analysis":
        st.subheader("🔗 Feature Correlation Analysis")
        num_cols, _ = numeric_and_categorical(df)
        
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns for correlation analysis.")
        else:
            # Add churn probability to correlation
            corr_df = df[num_cols].copy()
            corr_df['Churn_Probability'] = probabilities
            corr = corr_df.corr(numeric_only=True)
            
            fig = px.imshow(
                corr,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                origin='lower',
                title='Feature Correlation Matrix'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation with churn probability
            if 'Churn_Probability' in corr.index:
                churn_corr = corr['Churn_Probability'].drop('Churn_Probability').sort_values(ascending=False)
                st.subheader("Correlation with Churn Probability")
                fig2 = px.bar(
                    x=churn_corr.index,
                    y=churn_corr.values,
                    title='Feature Correlation with Churn Probability',
                    labels={'x': 'Feature', 'y': 'Correlation'},
                    color=churn_corr.values,
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Feature Importance":
        st.subheader("🎯 Feature Importance")
        utils, models_loaded = get_utils()
        
        available_tree_models = [name for name in ['random_forest', 'xgboost'] if models_loaded.get(name, False)]
        
        if not available_tree_models:
            st.info("Tree-based models (Random Forest or XGBoost) are required for feature importance visualization.")
        else:
            selected_model = st.selectbox(
                "Select Model",
                available_tree_models,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            try:
                importance_df = utils.get_feature_importance(selected_model, top_n=None)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Feature Importance - {selected_model.replace("_", " ").title()}',
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(importance_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error getting feature importance: {e}")
    
    elif viz_type == "High Risk Customers Analysis":
        st.subheader("⚠️ High Risk Customers Analysis")
        
        threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.05)
        high_risk = probabilities >= threshold
        high_risk_df = df[high_risk].copy()
        high_risk_df['Churn_Probability'] = probabilities[high_risk]
        
        st.metric("High Risk Customers", len(high_risk_df), delta=f"{(len(high_risk_df)/len(df))*100:.1f}%")
        
        if len(high_risk_df) > 0:
            # Analyze high risk customers
            if 'Geography' in df.columns:
                geo_dist = high_risk_df['Geography'].value_counts()
                fig = px.pie(
                    values=geo_dist.values,
                    names=geo_dist.index,
                    title='High Risk Customers by Geography'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Gender' in df.columns:
                gender_dist = high_risk_df['Gender'].value_counts()
                fig = px.bar(
                    x=gender_dist.index,
                    y=gender_dist.values,
                    title='High Risk Customers by Gender',
                    labels={'x': 'Gender', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Age' in df.columns:
                fig = px.histogram(
                    high_risk_df,
                    x='Age',
                    nbins=30,
                    title='Age Distribution of High Risk Customers',
                    labels={'Age': 'Age', 'count': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show table
            st.subheader("High Risk Customers Details")
            display_df = high_risk_df.sort_values('Churn_Probability', ascending=False)
            display_df['Churn_Probability'] = display_df['Churn_Probability'].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df.head(50), use_container_width=True)
        else:
            st.info(f"No customers found with churn probability >= {threshold:.0%}")

                