import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import microspot_util.plots as plots
import microspot_util.streamlit as mst
import microspot_util as msu

mst.page_setup()

mst.datainfo()

st.markdown("# Data Merging")

choose_input=st.selectbox("File upload:",["Use Selection in current Session","Upload Data"])

if choose_input=="Use Selection in current Session":

    data_list=[st.session_state["session_data"][data_id] for _,data_id in st.session_state["session_df"].loc[st.session_state["session_df"]["Select"]==True,"id"].items()]

    if len(data_list)>0: 
        mst.mergedata_loaded()
        st.dataframe(st.session_state["session_df"].loc[st.session_state["session_df"]["Select"]==True,"Name"],column_config={"Name":"Selected Datasets:"},use_container_width=True,hide_index=True)


elif choose_input=="Upload Data":
    
    upload_list=st.file_uploader("Upload all .csv files.","csv",accept_multiple_files=True,on_change=mst.mergedata_loaded)

    data_list=[pd.read_csv(item) for item in upload_list]


c1,c2=st.columns(2)

with c1:
    st.toggle("Add Retention-Time",key="addRT")
    t_0=st.number_input("Start Time [s]",value=0,disabled=not st.session_state["addRT"])
    test=st.button("Merge Data",disabled=st.session_state["mergedata_loaded"],type="primary",on_click=mst.merge_settings)

with c2:
    st.toggle("Serpentine Path",key="serpentine",disabled=not st.session_state["addRT"])
    d_t=st.number_input("Time per spot [s]",value=1,disabled=not st.session_state["addRT"])
    
if st.session_state["merge_state"]==True:
    merged_spots=[]
    for spotlist in data_list:
        merged_spots.extend(msu.spot.df_to_list(spotlist))
        merged_spots.sort(key=lambda x: x.row+(x.col/1000))

    st.dataframe(msu.spot.create_df(merged_spots))