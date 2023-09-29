import streamlit as st
import pandas as pd

states={"analyze":False,"session_data":pd.DataFrame(columns=["Data","Name","Select"]),"init_analysis":True,"img":None,"grid":None,"current_data":None,"edit":None,"change_warning":False}

def set_analyze_True():
    st.session_state["analyze"]=True

def page_setup():
    st.set_page_config(
        page_title="Microspot Reader",
        page_icon=r"assets\Logo_notext.png",
        initial_sidebar_state="auto",
        menu_items=None
        )
    
    # with st.sidebar:
    #     st.image(r"assets\logo_µspotreader.png")
    
    for name,state in states.items():
        if name not in st.session_state:
            st.session_state[name]=state

def apply_datachange():
    st.session_state["session_data"]=st.session_state["edit"]
    st.session_state["change_warning"]=False

def del_sessiondata():
    st.session_state["session_data"]=st.session_state["edit"]
    st.session_state["session_data"]=st.session_state["session_data"].loc[st.session_state["edit"]["Select"]==False]
    st.session_state["edit"]=st.session_state["edit"].loc[st.session_state["edit"]["Select"]==False]
    st.session_state["change_warning"]=False

def datachange_warning():
    st.session_state["change_warning"]=True

def datainfo():
    with st.sidebar:
        st.caption("Image data in current Session:")
        
        if st.session_state["change_warning"]==True:
            st.warning("Changes have not been applied yet!")

        st.session_state["edit"]=st.data_editor(st.session_state["session_data"],column_config={"Select":st.column_config.CheckboxColumn("Select",default=False),"Data":None},use_container_width=True,hide_index=True,on_change=datachange_warning)

        col1,col2=st.columns(2)
        with col2:
            st.button("Delete Selection",on_click=del_sessiondata,use_container_width=True)
        with col1:        
            st.button("Apply Changes",on_click=apply_datachange,use_container_width=True,type="primary")

def v_space(n, col=None):
    for _ in range(n):
        if col:
            col.write("")
        else:
            st.write("")

@st.cache_resource
def convert_df(df):
    return df.to_csv().encode("utf-8")