import sys
import platform
import streamlit as st
import pandas as pd
import plotly


def about_page():
    # Hero section
    st.title("✨ Data Analysis Template")
    st.caption("A fast, friendly Streamlit starter for uploading, exploring, visualizing, and exporting your data")

    # Quick overview cards (cleaner than metrics for non-numeric info)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        **📦 File Types**
        
        CSV • XLSX
        """)
    with c2:
        st.markdown("""
        **📊 Charts**
        
        Histogram • Box • Scatter • Heatmap
        """)
    with c3:
        st.markdown("""
        **🧭 Flow**
        
        Upload → Explore → Visualize → Export
        """)
    with c4:
        st.markdown("""
        **🧰 Libraries**
        
        Streamlit • Pandas • Plotly
        """)

    st.markdown("---")

    # Tabs for overview and guidance
    tab_overview, tab_quickstart, tab_features, tab_faq, tab_resources = st.tabs(
        ["Overview", "Quickstart", "Features", "FAQ", "Resources"]
    )

    with tab_overview:
        st.subheader("What this app does")
        st.write(
            """
            This template helps you move from raw data to quick insights in minutes.
            Load a dataset, inspect its structure, visualize patterns, and export results.
            """
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            #### Why you'll like it
            - Zero boilerplate: everything wired and ready
            - Clean, minimal UI with sensible defaults
            - Interactive visuals powered by Plotly
            - Extensible structure: add pages and features easily
            """)
        with c2:
            st.markdown("""
            #### Typical use-cases
            - Quick EDA on CSV/XLSX files
            - Exploring sample datasets (Iris, Tips, Gapminder)
            - Sharing lightweight data demos with teammates
            - Exporting filtered data for downstream tasks
            """)

    with tab_quickstart:
        st.subheader("Get started in 4 steps")
        st.info("Use the left sidebar to navigate between pages.")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown("""
            **1. 📥 Upload**
            
            CSV/XLSX or sample data
            """)
        with s2:
            st.markdown("""
            **2. 🔍 Explore**
            
            Schema, nulls, summary stats
            """)
        with s3:
            st.markdown("""
            **3. 📈 Visualize**
            
            Histogram, Box, Scatter, Heatmap
            """)
        with s4:
            st.markdown("""
            **4. 💾 Export**
            
            Full or filtered CSV
            """)
        st.caption("Tip: Adjust preview rows and plot options interactively on each page.")

    with tab_features:
        st.subheader("Feature highlights")
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("""
            - ✅ CSV/XLSX upload
            - ✅ Cached parsing for speed
            - ✅ Sample datasets (Iris, Tips, Gapminder)
            """)
        with f2:
            st.markdown("""
            - ✅ Schema table (dtypes, nulls)
            - ✅ Summary stats (numeric)
            - ✅ Session-state workflow
            """)
        with f3:
            st.markdown("""
            - ✅ Interactive charts
            - ✅ Correlation heatmap
            - ✅ Quick filter + CSV export
            """)

        with st.expander("Architecture at a glance", expanded=False):
            st.markdown("""
            - `front/app.py`: page layout, sidebar, and routing
            - `front/upload.py`: file upload and sample data handling
            - `front/explore.py`: schema and summary stats
            - `front/visual.py`: interactive plots
            - `front/export.py`: quick filter and CSV download
            - `front/state.py`: shared session state helpers
            """)

    with tab_faq:
        st.subheader("Frequently asked questions")
        with st.expander("What file formats are supported?"):
            st.write("CSV and Excel (XLSX/XLS).")
        with st.expander("Where do sample datasets come from?"):
            st.write("Plotly Express sample datasets (Iris, Tips, Gapminder).")
        with st.expander("Can I add more charts or pages?"):
            st.write("Yes—add modules under `front/` and wire them in `front/app.py`.")
        with st.expander("How is data stored during the session?"):
            st.write("In `st.session_state['df']` via helpers in `front/state.py`.")

    with tab_resources:
        st.subheader("Docs and references")
        colr1, colr2, colr3 = st.columns(3)
        with colr1:
            st.link_button("Streamlit Docs", "https://docs.streamlit.io/")
            st.link_button("Pandas Docs", "https://pandas.pydata.org/docs/")
        with colr2:
            st.link_button("Plotly Express", "https://plotly.com/python/plotly-express/")
            st.link_button("Streamlit Gallery", "https://streamlit.io/gallery")
        with colr3:
            st.link_button("Cheat Sheet", "https://docs.streamlit.io/library/cheatsheet")

        st.markdown("---")
        st.subheader("Environment")
        env_cols = st.columns(4)
        with env_cols[0]:
            st.caption("Python")
            st.code(sys.version.split(" ")[0])
        with env_cols[1]:
            st.caption("Streamlit")
            st.code(st.__version__)
        with env_cols[2]:
            st.caption("Pandas")
            st.code(pd.__version__)
        with env_cols[3]:
            st.caption("Plotly")
            st.code(plotly.__version__)

        st.caption(f"Platform: {platform.system()} {platform.release()}")

    # Gentle nudge
    st.markdown("---")
    cta1, cta2, cta3, cta4 = st.columns(4)
    cta1.success("Go to Upload →")
    cta2.info("Then Explore →")
    cta3.warning("Then Visualize →")
    cta4.success("Finally Export →")

    st.caption("Pro tip: You can always return here via the 'About' page in the sidebar.")
