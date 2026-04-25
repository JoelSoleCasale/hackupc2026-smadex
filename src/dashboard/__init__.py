import streamlit as st

from src.dashboard.pages.performance_explorer import render_performance_explorer


def render() -> None:
    st.set_page_config(
        page_title="Smadex Creative Intelligence",
        layout="wide",
        page_icon="📊",
    )
    pages = [
        st.Page(render_performance_explorer, title="Creative Performance", icon="📊"),
    ]
    pg = st.navigation(pages)
    pg.run()
