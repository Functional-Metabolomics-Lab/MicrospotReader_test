import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

import microspot_reader.plots as plots
import microspot_reader.streamlit as mst
import microspot_reader as msu

# Dictionaries to convert Row-Letters into Row-Numbers and vice versa (required for heatmap)
row_conv={"abcdefghijklmnopqrstuvwxyz"[i-1]: i for i in range(1,27)}
row_conv_inv={v:k for k,v in row_conv.items()}

# Initialize session-states and add basic design elements.
mst.page_setup()

# Add page specific utility to sidebar.
with st.sidebar:
    # Form to add merged data to the current session to be used later on.
    with st.form("Add to Session"):
        c1,c2=st.columns(2)
        
        # Name of data, is customizable by user
        with c2:
            data_name=st.text_input(
                "Name your Data",
                placeholder="Name your Data",
                label_visibility="collapsed"
            )

        # Adds the data to the current session, storage is explaned in the documentation of the add_mergedata function.            
        with c1:
            add=st.form_submit_button(
                "Add Prepared Data to Session",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state["merge_state"]
            )

        if add:
            
            if len(data_name)>0:
                mst.add_mergedata(data_name)
            
            else:
                # Warning message if no name has been entered.
                st.warning("Please enter a Name!")

    # Displays data that has been saved in the current session in tabular format.
    mst.datainfo()

st.markdown("# Data Merging and Preparation")

# Selection between custom file upload or selection from saved data in the current session
choose_input=st.selectbox(
    "File upload:",
    ["Use Selection in current Session","Upload Data"]
)

# Loading data for selection of stored session data.
if choose_input=="Use Selection in current Session":
    # Loading all selected spot-lists into a list
    data_list=[st.session_state["img_data"][data_id] for _,data_id in st.session_state["img_df"].loc[st.session_state["img_df"]["Select"]==True,"id"].items()]

    # If the List contains data, enable Start Data Preparation button and display the names of the spotlists.
    if len(data_list)>0: 
        st.session_state["mergedata_loaded"]=False
        
        st.dataframe(
            st.session_state["img_df"].loc[st.session_state["img_df"]["Select"]==True,"Name"],
            column_config={"Name":"Selected Datasets:"},
            use_container_width=True,
            hide_index=True
        )
    
    # Disable "Start Data Preparation" button if list does not contain data
    else:
        st.session_state["mergedata_loaded"]=True

# Loading uploaded data
elif choose_input=="Upload Data":
    # List of uploaded files.
    upload_list=st.file_uploader(
        "Upload all .csv files.",
        "csv",
        accept_multiple_files=True,
        on_change=mst.reset_merge
    )

    # convert .csv files to lists of spots and store in a list
    data_list=[msu.spot.df_to_list(pd.read_csv(item)) for item in upload_list]

    # Enable "Start Data Preparation" Button if list contains data
    if len(data_list)>0:
        st.session_state["mergedata_loaded"]=False
    
    # Disable "Start Data Preparation" Button if list is empty
    else:
        st.session_state["mergedata_loaded"]=True


# Settings for Data Preparation
with st.form("Data Preparation Settings"):
    
    c1,c2=st.columns(2)
    
    with c1:
        # Toggle the use of normalized data
        st.selectbox(
                "Column containing activity data:",
                ["norm_intensity","spot_intensity"],
                index=0,
                key="toggleNorm"
            )
        # Toggle to ignore spots of type "control" when adding the retention time. Set this to true if a row or column is used as control that was not spotted using the microspotter set-up.
        st.toggle(
            "Ignore controls when adding RT",
            value=True, 
            key="ignoreCtrl",
        )
        # Toggle serpentine sorting, if enabled spots are sorted in a serpentine pattern
        st.toggle(
            "Serpentine Path",
            key="serpentine",
            value=True,
        )

        st.markdown("####")

        # Button starting the data preparation process.
        dataprep=st.form_submit_button(
            "Start Data Preparation",
            disabled=st.session_state["mergedata_loaded"],
            type="primary",
            on_click=mst.merge_settings
        )

    with c2:
        # Input for the retention time at which spotting was started
        t_0=st.number_input(
            "Start Time [s]",
            value=0.0,
        )
        # Time each spot was eluted to.
        t_end=st.number_input(
            "End Time [s]",
            value=520.0,
        )

        sigma_smooth=st.number_input(
            "Sigma-Value for gaussian smoothing:",
            value=1
        )

# Initializes the merging process if the "Start Data Preparation" button was pressed
if dataprep is True:
    # Concatenates all spotlists found in data_list to one list
    merged_spots=[]
    for spotlist in data_list:
        merged_spots.extend(spotlist)
    
    # Extract information on first and last spot
    first_spot=merged_spots[0].row_name+str(merged_spots[0].col)
    last_spot=merged_spots[-1].row_name+str(merged_spots[-1].col)

    # Sorts the spots according to the settings
    msu.spot.sort_list(
        merged_spots,
        serpentine=st.session_state["serpentine"],
        inplace=True
    )

    # Create a dataframe from the spot-data
    df=msu.spot.create_df(merged_spots)
    
    # Annotation of all spots with a retention time 
    if st.session_state["ignoreCtrl"]==True:
        
        df.loc[df["type"]=="Sample","RT"]=np.linspace(
            t_0,
            t_end,
            num=len(df.loc[df["type"]=="Sample"])
        )
    
    elif st.session_state["ignoreCtrl"]==False:
        
        df.loc[df["type"] != None, "RT"]=np.linspace(
            t_0,
            t_end,
            num=len(df)
        )

    df=df.loc[df["RT"]>0].reset_index().copy()
    
    # baseline,level,df[st.session_state["toggleNorm"]]=msu.baseline_correction(
    #     array=df[st.session_state["toggleNorm"]],
    #     conv_lvl=0.001,
    #     conv_noise=0.00001,
    #     window_lvl=100,
    #     window_noise=5,
    #     poly_lvl=2,
    #     poly_noise=3
    # )

    # baseline correction using savitz-golay filter
    baseline,df["smoothed_int"]=msu.baseline_correction2(
        array=df[st.session_state["toggleNorm"]],
        conv_lvl=0.001,
        window_lvl=100,
        poly_lvl=1,
    )

    # curve smoothing using gaussian
    df["smoothed_int"]=gaussian_filter1d(
        input=df["smoothed_int"].to_numpy(),
        sigma=sigma_smooth
    )

    # stores current data in a session state
    st.session_state["current_merge"]=df

    # Get the grid-properties of the spotlist (required for heatmap).
    grid_props=msu.conv_gridinfo(first_spot,last_spot,row_conv)

    st.session_state["merge_results"]={
        "df":df,
        "grid":grid_props
    }

    st.session_state["merge_state"]=True

if st.session_state["merge_state"] is True:
    # 3 tabs if retention time is enabled
    t1,t2,t3=st.tabs(["Merged Data Table","Heatmap","Chromatogram"])
    
    with t3:
        # Plot a chromatogramm of spot intensities 
        fig,ax=plt.subplots()
        plots.plot_chromatogram(
            figure=fig,
            axs=ax,
            df=st.session_state["merge_results"]["df"],
            norm_data=st.session_state["toggleNorm"]=="norm_intensity"
        )
        st.pyplot(fig)

    with t1:
        # Display merged table
        st.dataframe(st.session_state["merge_results"]["df"],hide_index=True)
        
    with t2:
        if st.session_state["toggleNorm"]=="norm_intensity":
            cbar_name="Smoothed Normalized Intensity [a.u.]"
        else:
            cbar_name="Smoothed Spot Intensity [a.u.]"
        
        # display heatmap of merged data
        fig,ax=plt.subplots()
        plots.plot_heatmapv2(
            figure=fig,
            axs=ax,
            df=st.session_state["merge_results"]["df"],
            conv_dict=row_conv_inv,
            value_col="smoothed_int",
            colorbar_name=cbar_name,
            halo=any(st.session_state["merge_results"]["df"].halo>0)
        )
        st.pyplot(fig)

    # Download data
    table=mst.convert_df(st.session_state["merge_results"]["df"])
    st.download_button(
        label="Download Merged Data as .csv",
        data=table,
        mime="text/csv"
    )