import streamlit as st
import pandas as pd

states={"analyze":False,"session_data":{},"init_analysis":True,"img":None,"grid":None,"current_data":None,"edit":None,"change_warning":False,"mergedata_loaded":True,"merge_state":False,"session_df":pd.DataFrame(columns=["Name","Select","id"]),"session_id":0}

def set_analyze_True():
    st.session_state["analyze"]=True

def page_setup():
    st.set_page_config(
        page_title="Microspot Reader",
        page_icon=r"assets\Logo_notext.png",
        initial_sidebar_state="auto",
        menu_items=None
        )
    
    for name,state in states.items():
        if name not in st.session_state:
            st.session_state[name]=state

def add_sessiondata(data):
    st.session_state["session_data"][st.session_state["session_id"]]=st.session_state["current_data"]
    add_data=pd.Series({"Name":data,"Select":False,"id":st.session_state["session_id"]})
    st.session_state["session_df"]=pd.concat([st.session_state["session_df"],add_data.to_frame().T],ignore_index=True)
    st.session_state["session_id"]+=1

def apply_datachange():
    st.session_state["session_df"]=st.session_state["edit"]
    st.session_state["change_warning"]=False

def del_sessiondata():
    st.session_state["session_df"]=st.session_state["session_df"].loc[st.session_state["edit"]["Select"]==False]

    st.session_state["session_data"]={st.session_state["session_df"].loc[idx,"id"]: st.session_state["session_data"][st.session_state["session_df"].loc[idx,"id"]] for idx in st.session_state["session_df"].index}

    st.session_state["change_warning"]=False

def datachange_warning():
    st.session_state["change_warning"]=True

def datainfo():
    with st.sidebar:
        st.caption("Image data in current Session:")
        
        if st.session_state["change_warning"]==True:
            st.warning("Changes have not been applied yet!")

        st.session_state["edit"]=st.data_editor(st.session_state["session_df"],column_config={"Select":st.column_config.CheckboxColumn("Select",default=False),"id":None},use_container_width=True,hide_index=True,on_change=datachange_warning)

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

def mergedata_loaded():
    st.session_state["mergedata_loaded"]=False

def merge_settings():
    st.session_state["merge_state"]=True