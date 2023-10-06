import streamlit as st
import pandas as pd

states={"analyze":False,
        "img_data":{},
        "init_analysis":True,
        "img":None,
        "grid":None,
        "current_img":None,
        "edit_img":None,
        "change_warning":False,
        "mergedata_loaded":True,
        "merge_state":False,
        "img_df":pd.DataFrame(columns=["Name","Select","id"]),
        "session_id":0,
        "edit_merge":None,
        "merge_data":{},
        "merge_df":pd.DataFrame(columns=["Name","Select","id"]),"merge_data":{},
        "current_merge":None}

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

def add_imgdata(data_name):
    st.session_state["img_data"][st.session_state["session_id"]]=st.session_state["current_img"]
    add_data=pd.Series({"Name":data_name,"Select":False,"id":st.session_state["session_id"]})
    st.session_state["img_df"]=pd.concat([st.session_state["img_df"],add_data.to_frame().T],ignore_index=True)
    st.session_state["session_id"]+=1

def add_mergedata(data_name):
    st.session_state["merge_data"][st.session_state["session_id"]]=st.session_state["current_merge"]
    add_data=pd.Series({"Name":data_name,"Select":False,"id":st.session_state["session_id"]})
    st.session_state["merge_df"]=pd.concat([st.session_state["merge_df"],add_data.to_frame().T],ignore_index=True)
    st.session_state["session_id"]+=1
    
def apply_datachange():
    st.session_state["img_df"]=st.session_state["edit_img"]
    st.session_state["change_warning"]=False

def del_sessiondata():
    st.session_state["img_df"]=st.session_state["img_df"].loc[st.session_state["edit_img"]["Select"]==False]

    st.session_state["img_data"]={st.session_state["img_df"].loc[idx,"id"]: st.session_state["img_data"][st.session_state["img_df"].loc[idx,"id"]] for idx in st.session_state["img_df"].index}

    st.session_state["change_warning"]=False

def datachange_warning():
    st.session_state["change_warning"]=True

def datainfo():
    with st.sidebar:
        st.markdown("### Data in current Session")
        
        if st.session_state["change_warning"]==True:
            st.warning("Changes have not been applied yet!")

        st.caption("Image-Data")
        st.session_state["edit_img"]=st.data_editor(st.session_state["img_df"],column_config={"Select":st.column_config.CheckboxColumn("Select",default=False),"id":None},use_container_width=True,hide_index=True,on_change=datachange_warning,key="1")

        st.caption("Merged Data")
        st.session_state["edit_merge"]=st.data_editor(st.session_state["merge_df"],column_config={"Select":st.column_config.CheckboxColumn("Select",default=False),"id":None},use_container_width=True,hide_index=True,on_change=datachange_warning,key="2")

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

def reset_merge():
    st.session_state["merge_state"]=False