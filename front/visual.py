import streamlit as st
import plotly.express as px
from explore import numeric_and_categorical

def visualize_data():
    st.header("Visualize")
    df = st.session_state.df
    if df.empty:
        st.warning("No data loaded yet. Go to 'Upload' to load a dataset.")
    else:
        num_cols, cat_cols = numeric_and_categorical(df)
        plot_type = st.selectbox("Plot type", ["Histogram", "Box", "Scatter", "Correlation heatmap"])

        if plot_type == "Histogram":
            if not num_cols:
                st.info("No numeric columns available.")
            else:
                x = st.selectbox("Column", num_cols)
                color = st.selectbox("Color (optional)", [None] + cat_cols)
                bins = st.slider("Bins", 5, 100, 30)
                fig = px.histogram(df, x=x, color=color if color else None, nbins=bins, marginal="rug")
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Box":
            if not num_cols:
                st.info("No numeric columns available.")
            else:
                y = st.selectbox("Y (numeric)", num_cols)
                x = st.selectbox("X (category, optional)", [None] + cat_cols)
                color = st.selectbox("Color (optional)", [None] + cat_cols)
                fig = px.box(df, x=x if x else None, y=y, color=color if color else None, points="outliers")
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Scatter":
            if len(num_cols) < 2:
                st.info("Need at least two numeric columns for a scatter plot.")
            else:
                x = st.selectbox("X", num_cols, index=0)
                y = st.selectbox("Y", [c for c in num_cols if c != x], index=0)
                color = st.selectbox("Color (optional)", [None] + cat_cols)
                size = st.selectbox("Size (optional)", [None] + num_cols)
                fig = px.scatter(
                    df, x=x, y=y,
                    color=color if color else None,
                    size=size if size else None,
                    trendline="ols" if st.checkbox("Add trendline") else None,
                )
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Correlation heatmap":
            if len(num_cols) < 2:
                st.info("Need at least two numeric columns for a correlation heatmap.")
            else:
                corr = df[num_cols].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", origin="lower")
                st.plotly_chart(fig, use_container_width=True)

                