import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import microspot_util as msu
import microspot_util.streamlit as mst
import microspot_util.plots as plots

# Initialize session-states and add basic design elements.
mst.page_setup()

# Add page specific utility to sidebar.
with st.sidebar:
    # Reset the Image analysis to start over again when results are displayed
    if st.button("Start New Analysis!",use_container_width=True,type="primary"):
        st.session_state["analyze"]=False
        st.session_state["current_img"]=None

    # Form to add spot-data from an image to the current session to be used later on.
    with st.form("Add to Session"):
        c1,c2=st.columns(2)
        
        # Name of data, is customizable by user
        with c2:
            data_name=st.text_input("Name your Data",placeholder="Name your Data",label_visibility="collapsed")

        # Adds the data to the current session, storage is explaned in the documentation of the add_imgdata function.       
        with c1:
            add=st.form_submit_button("Add Image-Data to Session",type="primary",use_container_width=True,disabled=not st.session_state["analyze"])

        if add:
            if len(data_name)>0:
                mst.add_imgdata(data_name)
            else:
                # Warning message if no name has been entered.
                st.warning("Please enter a Name!")        

# Displays data that has been saved in the current session in tabular format.
mst.datainfo()

st.markdown("# Image Analysis")

# Dictionaries to convert Row-Letters into Row-Numbers and vice versa.
row_conv={"abcdefghijklmnopqrstuvwxyz"[i-1]: i for i in range(1,27)}
row_conv_inv={v:k for k,v in row_conv.items()}

# If anaylsis has not been initialized, shows image preparation steps.
if st.session_state["analyze"]==False:
    # Selection between custom image upload or an example.
    choose_input=st.selectbox("File upload:",["Upload Image","Example for Testing"])

    # Example image
    if choose_input=="Example for Testing":
        inputfile=Path(r"test_images/edge_halo.tif")

    # File uploader for custom image files
    elif choose_input=="Upload Image":
        inputfile=st.file_uploader("Upload Microspot Image",["png","jpg","tif"])
    
    # after the inputfile was uploaded, show settings menu
    if inputfile:
        st.markdown("## Image to be analyzed")
        
        col1,col2=st.columns([0.6,0.4])
        
        # Displays the available Settings
        with col2:
            st.caption("Settings:")
            # Toggle the inversion of the grayscale image.
            invert=st.toggle("Invert grayscale Image",value=True)
            # Set the indexing for the first and last spot: Required for the calculation of grid parameters.
            first_spot=st.text_input("Index of First Spot",placeholder="A1")
            last_spot=st.text_input("Index of Last Spot",placeholder="P20")
            # Enables or disables the detection of Halos
            halo_toggle=st.toggle("Enable Halo detection",value=True,key="halo_toggle")
            
            # Enables start analysis button if all required settings are set and calculates grid information.
            if first_spot and last_spot:
                st.session_state["init_analysis"]=False
                st.session_state["grid"]=msu.conv_gridinfo(first_spot,last_spot,row_conv)
            
                # Labels the selected rows and columns as controls. All other spots are labeled as Samples
                ctrl_rows=st.multiselect('Select Rows to be labeled as "control"',[row_conv_inv[i].upper() for i in range(st.session_state["grid"]["rows"]["bounds"][0],st.session_state["grid"]["rows"]["bounds"][1]+1)],key="ctrl_rows")
                ctrl_cols=st.multiselect('Select Columns to be labeled as "control"',list(range(st.session_state["grid"]["columns"]["bounds"][0],st.session_state["grid"]["columns"]["bounds"][1]+1)),key="ctrl_cols")

            # Disables start analysis button and sends out warning if not all settings have been set
            else:
                st.session_state["init_analysis"]=True
                st.warning("Please enter Spot Indices!")

        # Load and prepare raw image file.
        raw_img=msu.prep_img(inputfile, invert=invert)
        st.session_state["img"]=raw_img
        st.toast("Image prepared Successfully!")

        with col1:
            # Display the grayscale image using the "viridis" colormap.
            fig,ax=plt.subplots()
            ax.imshow(raw_img)
            ax.axis("off")  
            st.pyplot(fig)

        
        with st.expander("Advanced Settings!"):
            st.session_state["adv_settings"]={"init_det":{},}
            
            st.markdown("__Initial Spot-Detection__")
            
            c1,c2=st.columns(2)
            with c1:
                st.session_state["adv_settings"]["init_det"]["low_rad"]=st.number_input("Smallest tested radius:",value=20,step=1)
                st.session_state["adv_settings"]["init_det"]["sigma"]=st.number_input("Sigma-Value for Gaussian Blur:",min_value=1,max_value=20,step=1,value=10)
                st.session_state["adv_settings"]["init_det"]["low_edge"]=st.number_input("Edge-Detection low threshold:",value=0.001)
                st.session_state["adv_settings"]["init_det"]["high_edge"]=st.number_input("Edge-Detection high threshold:",value=0.001)

            with c2:
                st.session_state["adv_settings"]["init_det"]["high_rad"]=st.number_input("Largest tested radius:",value=30,step=1)
                st.session_state["adv_settings"]["init_det"]["x_dist"]=st.number_input("Minimum x-distance between spots:",value=70,step=1)
                st.session_state["adv_settings"]["init_det"]["y_dist"]=st.number_input("Minimum y-distance between spots:",value=70,step=1)
                st.session_state["adv_settings"]["init_det"]["thresh"]=st.number_input("Spot-Detection Threshold:",value=0.3)

            st.divider()

        with col1:
            # Start the image processing algorithm. Only activated if all settings have been set
            st.button("Start Analysis!",type="primary",disabled=st.session_state["init_analysis"],on_click=mst.set_analyze_True, use_container_width=True)

# Initiates Analysis and displays results if Starts Analysis button has been pressed.
if st.session_state["analyze"]==True:
    # Inital spot-detection.
    init_spots=msu.spot.detect(gray_img=st.session_state["img"],
                               spot_nr=st.session_state["grid"]["spot_nr"],
                               canny_sig=st.session_state["adv_settings"]["init_det"]["sigma"],
                               canny_lowthresh=st.session_state["adv_settings"]["init_det"]["low_edge"],
                               canny_highthresh=st.session_state["adv_settings"]["init_det"]["high_edge"],
                               hough_minx=st.session_state["adv_settings"]["init_det"]["x_dist"],
                               hough_miny=st.session_state["adv_settings"]["init_det"]["y_dist"],
                               hough_thresh=st.session_state["adv_settings"]["init_det"]["thresh"],
                               small_rad=st.session_state["adv_settings"]["init_det"]["low_rad"],
                               large_rad=st.session_state["adv_settings"]["init_det"]["high_rad"],
                               )
        
    # Create an empty image and draw a dot for each detected spot.
    dot_img=np.zeros(st.session_state["img"].shape)
    for i_spot in init_spots: 
        i_spot.draw_spot(dot_img,255,5)

    # Detection of gridlines.
    gridlines=msu.gridline.detect(img=dot_img, 
                                  max_tilt=5,
                                  min_dist=80,
                                  threshold=0.2
                                  )
    hor_line=[line for line in gridlines if line.alignment=="hor"]
    vert_line=[line for line in gridlines if line.alignment=="vert"]

    # Calculate the intersections of all horizontal lines with all vertical lines.
    grid_points=[]
    for h_l in hor_line:
        for v_l in vert_line:
            grid_points.append(v_l.intersect(h_l))

    # Initializing corrected spotlist
    corr_spots=[]

    # Loop over all spots and gridpoints
    for s_point in init_spots:
        dist_list=[]
        for g_point in grid_points:

            # Calculate the distance between the points and append it to the running list.
            pointdist=g_point.eval_distance(s_point.x,s_point.y)
            dist_list.append(pointdist)
        
        # If the distance between the current spot and any gridpoint is <= it is accepted as correct.
        if min(dist_list)<=10:
            corr_spots.append(s_point)

    # Loop over all gridpoints and backfill the ones that are not associated with a spot.
    for g_point in grid_points:
        if g_point.min_dist>10:
            msu.spot.backfill(corr_spots,g_point.x,g_point.y)

    # Assigns each spot a place on the grid.
    sort_spots=msu.spot.sort_grid(corr_spots,
                                row_conv=row_conv_inv,
                                row_start=st.session_state["grid"]["rows"]["bounds"][0],
                                col_start=st.session_state["grid"]["columns"]["bounds"][0])

    # Calcualte the spot intensity and label controls
    for s in sort_spots:
        s.get_intensity(st.session_state["img"])
        if s.row_name in st.session_state["ctrl_rows"] or s.col in st.session_state["ctrl_cols"]:
            s.type="Control"

    if st.session_state["halo_toggle"]==True:
        # Detect Halos using the halo.detect method.
        halos=msu.halo.detect(img=st.session_state["img"],
                              canny_sig=3.52941866,
                              canny_lowthresh=44.78445877,
                              canny_highthresh=44.78445877,
                              hough_minx=70,
                              hough_miny=70,
                              hough_thresh=0.38546213,
                              )

        # Assign halos to their spot.
        for s in sort_spots:
            s.assign_halo(halos)

    # If controls are present, normalize the spot intensities 
    if len(st.session_state["ctrl_rows"]) != 0 or len(st.session_state["ctrl_cols"]) != 0:
        msu.spot.normalize(sort_spots)

    st.markdown("## Results")

    # Tabs for all Results that are displayed
    tab1,tab2,tab3,tab4=st.tabs(["Image","Table","Heatmap", "Grid"])

    # saves spot-list in a session state
    st.session_state["current_img"]=sort_spots
    
    # Turns spotlist to df for visualization and download.
    df=msu.spot.create_df(sort_spots)
    
    # Displays image with main results.
    with tab1:
        col1,col2=st.columns([0.6,0.4])
        with col1:
            fig,ax=plt.subplots()
            plots.plot_result(fig,ax,st.session_state["img"],df,st.session_state["grid"])        
            st.pyplot(fig)

    # Displays the Table containing all information on the spots
    with tab2:
        st.dataframe(df)

    # Displays a heatmap of spot-intensities
    with tab3:
        col1,col2=st.columns([0.6,0.4])
        with col1:
            # Display Image and corresponding Heatmap
            fig,ax=plt.subplots()
            plots.plot_heatmap(fig,ax,df,st.session_state["grid"])
            st.pyplot(fig)
    
    # Displays the detected grid.
    with tab4:
        col1,col2=st.columns(2)
        with col1:
            # Display the grid.
            fig,ax=plt.subplots()
            plots.plot_grid(fig,ax,st.session_state["img"],hor_line+vert_line)
            st.pyplot(fig)
            
        with col2: 
            st.markdown("## Detected Grid")
            mst.v_space(1)
            st.markdown("Please check whether the gridlines match the Spots!")
            st.markdown("A faulty grid leads to errors during spot detection and can influence the results negatively. The most frequent reason for faulty grids is a noisy background in the submitted image.")

    # Converts the dataframe to a .csv and adds a download button.
    table=mst.convert_df(df)
    st.download_button(label="Download Spot-Data as .csv",data=table,mime="text/csv")
