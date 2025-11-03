import streamlit as st

def about_page():
    st.title("About This App")
    st.markdown("""
    ## Customer Churn Prediction Application
    
    This application uses machine learning to predict customer churn probability. It analyzes customer data 
    and provides insights to help identify at-risk customers and implement retention strategies.
    
    ### Features:
    - **📤 Upload**: Upload CSV or Excel files with customer data
    - **🔍 Explore**: View customer data with churn predictions in an interactive table
    - **📊 Visualize**: Generate comprehensive charts and statistics on churn prediction
    - **📥 Export**: Download results with predictions for further analysis
    
    ### Machine Learning Models:
    The application uses three trained models:
    - **Logistic Regression**: A linear model providing interpretable results
    - **Random Forest**: An ensemble tree-based method capturing non-linear relationships
    - **XGBoost**: A gradient boosting method known for high performance
    
    ### Required Data Format:
    Your dataset should contain the following columns:
    - `CreditScore`: Customer's credit score
    - `Geography`: Customer's location (France, Germany, Spain)
    - `Gender`: Customer's gender (Male, Female)
    - `Age`: Customer's age
    - `Tenure`: Number of years as a customer
    - `Balance`: Account balance
    - `NumOfProducts`: Number of products
    - `HasCrCard`: Has credit card (0 or 1)
    - `IsActiveMember`: Active membership status (0 or 1)
    - `EstimatedSalary`: Customer's estimated salary
    
    ### Technologies Used:
    - **Streamlit**: Web interface framework
    - **Scikit-learn**: Machine learning library (Logistic Regression, Random Forest)
    - **XGBoost**: Gradient boosting framework
    - **Pandas**: Data manipulation and analysis
    - **Plotly**: Interactive data visualization
    
    ### How to Use:
    1. Navigate to the **Upload** page and upload your customer dataset (CSV or Excel)
    2. Go to the **Explore** page to see predictions and view customer data with churn probabilities
    3. Use the **Visualize** page to explore various charts and statistics:
       - Churn distribution
       - Churn by geography and gender
       - Age distribution analysis
       - Feature importance
       - High-risk customer analysis
    4. Export your results from the **Export** page for further analysis
    
    ### Model Performance:
    The models have been trained on historical customer data and evaluated using:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    
    **Note**: Models are loaded from the `models/` directory. Ensure model files are present for the application to work properly.
    """)