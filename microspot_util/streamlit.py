import streamlit as st
import pandas as pd

def page_setup():
    st.set_page_config(
        page_title="Microspot Reader",
        page_icon=r"assets\Logo_notext.png",
        initial_sidebar_state="auto",
        menu_items=None
        )
    
    if "analyze" not in st.session_state:
        st.session_state["analyze"]=False

    if "data" not in st.session_state:
        st.session_state["data"]=pd.DataFrame(columns=["Data","Name","Select"])

    with st.sidebar:
        with st.form("Data in Session"):
            st.markdown("## Data in Session:")
            st.session_state["data"]=st.data_editor(st.session_state["data"],column_config={"Select":st.column_config.CheckboxColumn("Select",default=False),"Data":None},use_container_width=True,hide_index=True,)

            c1,c2=st.columns(2)
            with c2:
                if st.form_submit_button("Delete Selection"):
                    st.session_state["data"]=st.session_state["data"].loc[st.session_state["data"]["Select"]==False]
                    st.rerun()

            with c1: 
                if st.form_submit_button("Submit Changes"):
                    st.session_state["data"]=st.session_state["data"]


def v_space(n, col=None):
    for _ in range(n):
        if col:
            col.write("")
        else:
            st.write("")

@st.cache_resource
def convert_df(df):
    return df.to_csv().encode("utf-8")