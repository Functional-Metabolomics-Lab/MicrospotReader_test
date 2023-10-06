import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import microspot_util.plots as plots
import microspot_util.streamlit as mst
import microspot_util as msu

mst.page_setup()

with st.sidebar:
    with st.form("Add to Session"):
        c1,c2=st.columns(2)
        with c2:
            data_name=st.text_input("Name your Data",placeholder="Name your Data",label_visibility="collapsed")
                    
        with c1:
            add=st.form_submit_button("Add Merged Data to Session",type="primary",use_container_width=True,disabled=not st.session_state["merge_state"])

        if add:
            if len(data_name)>0:
                mst.add_mergedata(data_name)
            else:
                st.warning("Please enter a Name!")

mst.datainfo()

st.markdown("# Data Merging")

choose_input=st.selectbox("File upload:",["Use Selection in current Session","Upload Data"])

if choose_input=="Use Selection in current Session":

    data_list=[st.session_state["img_data"][data_id] for _,data_id in st.session_state["img_df"].loc[st.session_state["img_df"]["Select"]==True,"id"].items()]

    if len(data_list)>0: 
        st.session_state["mergedata_loaded"]=False
        st.dataframe(st.session_state["img_df"].loc[st.session_state["img_df"]["Select"]==True,"Name"],column_config={"Name":"Selected Datasets:"},use_container_width=True,hide_index=True)
    
    else:
        st.session_state["mergedata_loaded"]=True


elif choose_input=="Upload Data":
    
    upload_list=st.file_uploader("Upload all .csv files.","csv",accept_multiple_files=True,on_change=mst.reset_merge)
    print(upload_list)

    data_list=[msu.spot.df_to_list(pd.read_csv(item)) for item in upload_list]

    # data_list=[msu.spot.create_df(pd.read_csv(item)) for item in upload_list]

    if len(data_list)>0:
        st.session_state["mergedata_loaded"]=False
    
    else:
        st.session_state["mergedata_loaded"]=True

c1,c2=st.columns(2)

with c1:
    st.toggle("Add Retention-Time",key="addRT",on_change=mst.reset_merge)
    t_0=st.number_input("Start Time [s]",value=0,disabled=not st.session_state["addRT"],on_change=mst.reset_merge)
    st.button("Merge Data",disabled=st.session_state["mergedata_loaded"],type="primary",on_click=mst.merge_settings)

with c2:
    st.toggle("Serpentine Path",key="serpentine",on_change=mst.reset_merge)
    d_t=st.number_input("Time per spot [s]",value=1,disabled=not st.session_state["addRT"],on_change=mst.reset_merge)
    
if st.session_state["merge_state"]==True:
    merged_spots=[]
    for spotlist in data_list:
        merged_spots.extend(spotlist)
    
    msu.spot.sort_list(merged_spots,serpentine=st.session_state["serpentine"],inplace=True)

    if st.session_state["addRT"]==True:
        msu.spot.annotate_RT(merged_spots,t_0,d_t)

    df=msu.spot.create_df(merged_spots)
    st.session_state["current_merge"]=df
    st.dataframe(df)
    
    table=mst.convert_df(df)
    st.download_button(label="Download Merged Data as .csv",data=table,mime="text/csv")