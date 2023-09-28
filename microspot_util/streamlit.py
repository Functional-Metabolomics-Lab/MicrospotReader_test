import streamlit as st

def page_setup():
    st.set_page_config(
        page_title="Microspot Reader",
        page_icon="logo_µspotreader.png",
        initial_sidebar_state="auto",
        menu_items=None
        )

def v_space(n, col=None):
    for _ in range(n):
        if col:
            col.write("")
        else:
            st.write("")