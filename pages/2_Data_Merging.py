import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import microspot_util.plots as plots
import microspot_util.streamlit as mst
import microspot_util as msu

# Initialize session-states and add basic design elements.
mst.page_setup()

# Add page specific utility to sidebar.
with st.sidebar:
    # Form to add merged data to the current session to be used later on.
    with st.form("Add to Session"):
        c1,c2=st.columns(2)
        
        # Name of data, is customizable by user
        with c2:
            data_name=st.text_input("Name your Data",placeholder="Name your Data",label_visibility="collapsed")

        # Adds the data to the current session, storage is explaned in the documentation of the add_mergedata function.            
        with c1:
            add=st.form_submit_button("Add Merged Data to Session",type="primary",use_container_width=True,disabled=not st.session_state["merge_state"])

        if add:
            if len(data_name)>0:
                mst.add_mergedata(data_name)
            else:
                # Warning message if no name has been entered.
                st.warning("Please enter a Name!")


# Displays data that has been saved in the current session in tabular format.
mst.datainfo()

st.markdown("# Data Merging")

# Selection between custom file upload or selection from saved data in the current session
choose_input=st.selectbox("File upload:",["Use Selection in current Session","Upload Data"])

# Loading data for selection of stored session data.
if choose_input=="Use Selection in current Session":
    # Loading all selected spot-lists into a list
    data_list=[st.session_state["img_data"][data_id] for _,data_id in st.session_state["img_df"].loc[st.session_state["img_df"]["Select"]==True,"id"].items()]

    # If the List contains data, enable Merge Data button and display the names of the spotlists.
    if len(data_list)>0: 
        st.session_state["mergedata_loaded"]=False
        st.dataframe(st.session_state["img_df"].loc[st.session_state["img_df"]["Select"]==True,"Name"],column_config={"Name":"Selected Datasets:"},use_container_width=True,hide_index=True)
    
    # Disable "Merge Data" button if list does not contain data
    else:
        st.session_state["mergedata_loaded"]=True

# Loading uploaded data
elif choose_input=="Upload Data":
    # List of uploaded files.
    upload_list=st.file_uploader("Upload all .csv files.","csv",accept_multiple_files=True,on_change=mst.reset_merge)

    # convert .csv files to lists of spots and store in a list
    data_list=[msu.spot.df_to_list(pd.read_csv(item)) for item in upload_list]

    # Enable "Merge Data" Button if list contains data
    if len(data_list)>0:
        st.session_state["mergedata_loaded"]=False
    
    # Disable "Merge Data" Button if list is empty
    else:
        st.session_state["mergedata_loaded"]=True

col1,col2,col3=st.columns(3)
c1,c2=st.columns(2)

# Settings for Data Merging
with col1:
    # Toggle the use of normalized data
    st.toggle("Use normalized Data",key="toggleNorm",on_change=mst.reset_merge)

with col2:
    # Toggle the annotation of all spots with a retention time
    st.toggle("Add Retention-Time",key="addRT",on_change=mst.reset_merge)

with c1:
    # Input for the retention time at which spotting was started
    t_0=st.number_input("Start Time [s]",value=0,disabled=not st.session_state["addRT"],on_change=mst.reset_merge)
    # Button starting the data merging process.
    st.button("Merge Data",disabled=st.session_state["mergedata_loaded"],type="primary",on_click=mst.merge_settings)

with col3:
    # Toggle serpentine sorting, if enabled spots are sorted in a serpentine pattern
    st.toggle("Serpentine Path",key="serpentine",on_change=mst.reset_merge)
with c2:
    # Time each spot was eluted to.
    t_end=st.number_input("End Time [s]",value=520,disabled=not st.session_state["addRT"],on_change=mst.reset_merge)

# Initializes the merging process if the "Merge Data" button was pressed
if st.session_state["merge_state"]==True:
    # Concatenates all spotlists found in data_list to one list
    merged_spots=[]
    for spotlist in data_list:
        merged_spots.extend(spotlist)
    
    # Extract information on first and last spot
    first_spot=merged_spots[0].row_name+str(merged_spots[0].col)
    last_spot=merged_spots[-1].row_name+str(merged_spots[-1].col)

    # Sorts the spots according to the settings
    msu.spot.sort_list(merged_spots,serpentine=st.session_state["serpentine"],inplace=True)

    # Annotation of all spots with a retention time if enabled
    if st.session_state["addRT"]==True:
        msu.spot.annotate_RT(merged_spots,t_0,t_end)

    # creates a dataframe for download and visualization
    df=msu.spot.create_df(merged_spots)
    # stores current data in a session state
    st.session_state["current_merge"]=df

    # Dictionaries to convert Row-Letters into Row-Numbers and vice versa (required for heatmap)
    row_conv={"abcdefghijklmnopqrstuvwxyz"[i-1]: i for i in range(1,27)}
    row_conv_inv={v:k for k,v in row_conv.items()}

    # Get the grid-properties of the spotlist (required for heatmap).
    grid_props=msu.conv_gridinfo(first_spot,last_spot,row_conv)

    # Creates a Tab-View of the results
    if st.session_state["addRT"]==True:
        # 3 tabs if retention time is enabled
        t1,t2,t3=st.tabs(["Merged Data Table","Heatmap","Chromatogram"])
        
        with t3:
            # Plot a chromatogramm of spot intensities 
            fig,ax=plt.subplots()
            plots.plot_chromatogram(fig,ax,df,norm_data=st.session_state["toggleNorm"])
            st.pyplot(fig)

    else:
        # 2 tabs if retention time is disabled
        t1,t2=st.tabs(["Merged Data Table","Heatmap"])

    with t1:
        # Display merged table
        st.dataframe(df)

    with t2:
        # display heatmap of merged data
        fig,ax=plt.subplots()
        plots.plot_heatmap(fig,ax,df,grid_props,norm_data=st.session_state["toggleNorm"])
        st.pyplot(fig)

    # Download data
    table=mst.convert_df(df)
    st.download_button(label="Download Merged Data as .csv",data=table,mime="text/csv")