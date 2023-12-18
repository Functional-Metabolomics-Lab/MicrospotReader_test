import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import microspot_util as msu
import microspot_util.streamlit as mst
import microspot_util.plots as plots
import skimage

# Initialize session-states and add basic design elements.
mst.page_setup()
st.session_state["merge_state"]=False

# Add page specific utility to sidebar.
with st.sidebar:
    # Reset the Image analysis to start over again when results are displayed
    if st.button("Start New Analysis!",use_container_width=True,type="primary"):
        st.session_state["analyze"]=False
        st.session_state["current_img"]=None
        st.session_state["false_pos"]=[]

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


# Selection between custom image upload or an example.
choose_input=st.selectbox("File upload:",["Upload Image","Example for Testing"],on_change=mst.set_analyze_False)

# Example image
if choose_input=="Example for Testing":
    inputfile=Path(r"test_images/standard_mix.tif")

# File uploader for custom image files
else:
    inputfile=st.file_uploader("Upload Microspot Image",["png","jpg","tif"],on_change=mst.set_analyze_False)

# after the inputfile was uploaded, show settings menu
if inputfile:
    st.markdown("## Image to be analyzed")
    
    img_container=st.container()

    col1,col2=st.columns(2)
    
    # Displays the available Settings
    with col1:
        st.markdown("⚙️ ***Settings***")

        # Set the indexing for the first spot: Required for the calculation of grid parameters.
        first_spot=st.text_input("Index of First Spot",placeholder="A1",on_change=mst.set_analyze_False)
    
    with col2:
        # Toggle the inversion of the grayscale image.
        invert=st.toggle("Invert grayscale Image",value=True,on_change=mst.set_analyze_False)

        # Set the indexing for the last spot: Required for the calculation of grid parameters.
        last_spot=st.text_input("Index of Last Spot",placeholder="L20",on_change=mst.set_analyze_False)
    
    with st.form("Settings", border=False): 
        
        col1, col2 = st.columns(2)
    
        # Enables start analysis button if all required settings are set and calculates grid information.
        if first_spot and last_spot:
            st.session_state["init_analysis"]=False
            st.session_state["grid"]=msu.conv_gridinfo(first_spot,last_spot,row_conv)

            with col1: 
                # Lists containing all rows, columns and indices:
                rowlist=[row_conv_inv[i].upper() for i in range(
                    st.session_state["grid"]["rows"]["bounds"][0],
                    st.session_state["grid"]["rows"]["bounds"][1]+1
                    )]
                columnlist=list(
                    range(
                        st.session_state["grid"]["columns"]["bounds"][0],
                        st.session_state["grid"]["columns"]["bounds"][1]+1
                    ))
                idxlist=[r+str(c) for r in rowlist for c in columnlist]

                # Marks selected rows as not containing samples.
                st.session_state["ctrl_rows"]=st.multiselect(
                    'Select rows that do not contain sample:',
                    rowlist
                    )
                
                st.session_state["ctrl_spots"]= st.multiselect(
                    "Select spots to be used as controls:",
                    idxlist
                )
            
            with col2:
                # Marks selected columns as not containing samples
                st.session_state["ctrl_cols"]=st.multiselect(
                    'Select columns that do not contain sample:',
                    columnlist
                    )

                mst.v_space(2)
                st.session_state["norm_toggle"]=st.toggle(
                    "Normalize data using controls.",
                    value=True
                )

        # Disables start analysis button and sends out warning if not all settings have been set
        else:
            st.session_state["init_analysis"]=True
            st.warning("Please enter Spot Indices!")

        # Expander containing the important settings for initial spot detection.
        with st.expander("**Spot detection**",expanded=True):

            col1,col2=st.columns(2)
            with col1:
                st.session_state["adv_settings"]["init_det"]["low_rad"]=st.number_input(
                    "Smallest tested radius *[in pixels]*:",
                    value=20,
                    step=1,
                    min_value=1
                    )

                st.session_state["adv_settings"]["init_det"]["x_dist"]=st.number_input(
                    "Minimum x-distance between spots *[in pixels]*:",
                    value=70,
                    step=1,
                    min_value=0
                    )

            with col2:
                st.session_state["adv_settings"]["init_det"]["high_rad"]=st.number_input(
                    "Largest tested radius *[in pixels]*:",
                    value=30,
                    step=1,
                    min_value=1
                    )

                st.session_state["adv_settings"]["init_det"]["y_dist"]=st.number_input(
                    "Minimum y-distance between spots *[in pixels]*:",
                    value=70,
                    step=1,
                    min_value=0
                    )
        
        # Expander containing the important settings for spot correction.
        with st.expander("**Spot correction**",expanded=True):
            c1,c2=st.columns(2)
            with c1:
                st.session_state["adv_settings"]["grid_det"]["min_dist"]=st.number_input(
                    "Minimum distance between grid-points *[in pixels]*:",
                    value=80,
                    step=1,
                    min_value=0
                    )

            with c2:
                st.session_state["adv_settings"]["spot_misc"]["acceptance"]=st.number_input(
                    "Max. spot-distance from grid-points *[in pixels]*:",
                    value=10,
                    min_value=1,
                    step=1
                    )

        # Expander containing the important settings for halo detection.
        with st.expander("**Halo detection**",expanded=False):
            # Enables or disables the detection of Halos
            st.session_state["halo_toggle"]=st.toggle(
                "Enable Halo detection",
                value=False
                )

            c1,c2=st.columns(2)
            with c1:
                st.session_state["adv_settings"]["halo_det"]["low_rad"]=st.number_input(
                    "Smallest tested radius *[in pixels]*:",
                    value=40,
                    step=1,
                    min_value=1,
                    )

                st.session_state["adv_settings"]["halo_det"]["x_dist"]=st.number_input(
                    "Minimum x-distance between halos *[in pixels]*:",
                    value=70,
                    step=1,
                    min_value=0,
                    )
                
                st.session_state["adv_settings"]["halo_det"]["tog_scale"]=st.selectbox(
                    "Select method for scaling halos to data:",
                    ["Scale to normalized Data", "Scale to raw Data","Do not scale to any Data"]
                    )

            with c2:
                st.session_state["adv_settings"]["halo_det"]["high_rad"]=st.number_input(
                    "Largest tested radius *[in pixels]*:",
                    value=100,
                    step=1,
                    min_value=1,
                    )
                
                st.session_state["adv_settings"]["halo_det"]["y_dist"]=st.number_input(
                    "Minimum y-distance between halos *[in pixels]*:",
                    value=70,
                    step=1,
                    min_value=0,
                    )
                
                st.session_state["adv_settings"]["halo_det"]["scaling"]=st.number_input(
                    "Scaling Factor:",
                    value=25.0,
                    min_value=0.0,
                    )

        # Expander containing advanced settings unlikely to be changed by user.
        with st.expander("❗ Advanced Settings", expanded=False):          
            
            st.markdown("__Initial Spot-Detection__")
            c1,c2=st.columns(2)
            with c1:
                st.session_state["adv_settings"]["init_det"]["sigma"]=st.number_input(
                    "Sigma-value for gaussian blur:",
                    min_value=1,
                    max_value=20,
                    step=1,
                    value=10
                    )
                
                st.session_state["adv_settings"]["init_det"]["low_edge"]=st.number_input(
                    "Edge-detection low threshold:",
                    value=0.001,
                    min_value=0.000000,
                    format="%f"
                    )

            with c2:
                st.session_state["adv_settings"]["init_det"]["thresh"]=st.number_input(
                    "Spot-detection threshold:",
                    value=0.3,
                    min_value=0.0,
                    format="%f"
                    )
                
                st.session_state["adv_settings"]["init_det"]["high_edge"]=st.number_input(
                    "Edge-detection high threshold:",
                    value=0.001,
                    min_value=0.000000,
                    format="%f"
                    )

            st.divider()

            st.markdown("__Grid-Detection__")
            c1,c2=st.columns(2)
            with c1:
                st.session_state["adv_settings"]["grid_det"]["tilt"]=st.number_input(
                    "Maximum tilt of grid *[in degrees]*:",
                    value=5,
                    step=1,
                    min_value=0
                    )

            with c2:
                st.session_state["adv_settings"]["grid_det"]["thresh"]=st.number_input(
                    "Threshold for line-detection:",
                    value=0.2,
                    min_value=0.0
                    )

            st.divider()

            st.markdown("__Spot Correction and Intensity Evaluation__")

            st.session_state["adv_settings"]["spot_misc"]["invert_int"]=st.toggle(
                "Invert image for calculation of intensity",
                value=False
            )

            st.session_state["adv_settings"]["spot_misc"]["int_rad"]=st.number_input(
                "Disk-Radius for spot-intensity calculation *[in pixels]*:",
                value=0,
                min_value=0,
                step=1
                )
            
            st.divider()

            st.markdown("__Halo-Detection__")
            c1,c2=st.columns(2)
            with c1:
                st.session_state["adv_settings"]["halo_det"]["thresh"]=st.number_input(
                    "Halo-detection threshold:",
                    value=0.2,
                    min_value=0.0,
                    )
                
                st.session_state["adv_settings"]["halo_det"]["disk"]=st.number_input(
                    "Disk radius for morphological dilation *[in pixels]*:",
                    value=3,
                )

            with c2:
                st.session_state["adv_settings"]["halo_det"]["min_obj"]=st.number_input(
                    "Minimum Object Size *[in pixels]*:",
                    value=800,
                    min_value=0,
                    step=1,
                    )
            
            # Old halo detection settings:

            # st.session_state["adv_settings"]["halo_det"]["high_rad"]=st.number_input("Largest tested radius:",value=70,step=1,min_value=1,disabled=not st.session_state["halo_toggle"])
            # st.session_state["adv_settings"]["halo_det"]["thresh"]=st.number_input("Spot-detection threshold:",value=0.38546213,min_value=0.0,disabled=not st.session_state["halo_toggle"])
            # st.session_state["adv_settings"]["halo_det"]["sigma"]=st.number_input("Sigma-value for gaussian blur:",min_value=1.0,max_value=20.0,value=3.52941866,disabled=not st.session_state["halo_toggle"])
            # st.session_state["adv_settings"]["halo_det"]["low_edge"]=st.number_input("Edge-detection low threshold:",value=44.78445877,min_value=0.0,disabled=not st.session_state["halo_toggle"])
            # st.session_state["adv_settings"]["halo_det"]["high_edge"]=st.number_input("Edge-detection high threshold:",value=44.78445877,min_value=0.0,disabled=not st.session_state["halo_toggle"])

        with img_container:
            # Load and prepare raw image file.
            raw_img=msu.prep_img(inputfile, invert=invert)
            st.session_state["img"]=raw_img
            st.toast("Image prepared Successfully!")

            # Display the grayscale image using the "viridis" colormap.
            fig,ax=plt.subplots()
            img_plot=ax.imshow(raw_img)
            fig.colorbar(img_plot,shrink=0.5,label="Grayscale-Value")
            ax.axis("off")  
            st.pyplot(fig)

        # Start the image processing algorithm. Only activated if all settings have been set
        if st.form_submit_button("Start Analysis!",type="primary",disabled=st.session_state["init_analysis"], use_container_width=True):
            st.session_state["analyze"] = True

# Initiates Analysis and displays results if Starts Analysis button has been pressed.
if st.session_state["analyze"] is True:

    st.markdown("## Results")

    # Inital spot-detection.
    init_spots,test=msu.spot.detect(
        gray_img=st.session_state["img"],
        spot_nr=st.session_state["grid"]["spot_nr"],
        canny_sig=st.session_state["adv_settings"]["init_det"]["sigma"],
        canny_lowthresh=st.session_state["adv_settings"]["init_det"]["low_edge"],
        canny_highthresh=st.session_state["adv_settings"]["init_det"]["high_edge"],
        hough_minx=st.session_state["adv_settings"]["init_det"]["x_dist"],
        hough_miny=st.session_state["adv_settings"]["init_det"]["y_dist"],
        hough_thresh=st.session_state["adv_settings"]["init_det"]["thresh"],
        small_rad=st.session_state["adv_settings"]["init_det"]["low_rad"],
        large_rad=st.session_state["adv_settings"]["init_det"]["high_rad"],
        troubleshoot=True
        )
    
    avg_spotradius=np.mean([s.rad for s in init_spots])

    # Create an empty image and draw a dot for each detected spot.
    dot_img=np.zeros(st.session_state["img"].shape)
    for i_spot in init_spots: 
        i_spot.draw_spot(dot_img,255,5)

    # Detection of gridlines.
    gridlines=msu.gridline.detect(
        img=dot_img, 
        max_tilt=st.session_state["adv_settings"]["grid_det"]["tilt"],
        min_dist=st.session_state["adv_settings"]["grid_det"]["min_dist"],
        threshold=st.session_state["adv_settings"]["grid_det"]["thresh"]
        )
    # Assigning each line to horizontal or vertical.
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
        if min(dist_list)<=st.session_state["adv_settings"]["spot_misc"]["acceptance"]:
            corr_spots.append(s_point)

    # Loop over all gridpoints and backfill the ones that are not associated with a spot.
    for g_point in grid_points:
        if g_point.min_dist>st.session_state["adv_settings"]["spot_misc"]["acceptance"]:
            msu.spot.backfill(
                corr_spots,
                g_point.x,
                g_point.y,
                avg_spotradius
            )

    # Assigns each spot a place on the grid.
    sort_spots=msu.spot.sort_grid(
        corr_spots,
        row_conv=row_conv_inv,
        row_start=st.session_state["grid"]["rows"]["bounds"][0],
        col_start=st.session_state["grid"]["columns"]["bounds"][0]
    )

    if st.session_state["adv_settings"]["spot_misc"]["invert_int"] is True:
        st.session_state["img"]=skimage.util.invert(st.session_state["img"])

    # Calculate the spot intensity and label controls
    for s in sort_spots:
        s.get_intensity(
            st.session_state["img"],
            st.session_state["adv_settings"]["spot_misc"]["int_rad"]
        )
        if s.row_name in st.session_state["ctrl_rows"] or s.col in st.session_state["ctrl_cols"]:
            s.type=np.nan
        
        if s.row_name+str(s.col) in st.session_state["ctrl_spots"]:
            s.type="Control"

    # If controls are present, normalize the spot intensities 
    if st.session_state["norm_toggle"] is True:
        msu.spot.normalize(sort_spots)
        st.session_state["norm"]=True
    
    else:
        st.session_state["norm"]=False

    if st.session_state["halo_toggle"] is True:
        # Detect Halos using the halo.detect method.
        halos=msu.halo.detect(
            img=st.session_state["img"],
            min_xdist=st.session_state["adv_settings"]["halo_det"]["x_dist"],
            min_ydist=st.session_state["adv_settings"]["halo_det"]["y_dist"],
            thresh=st.session_state["adv_settings"]["halo_det"]["thresh"],
            min_rad=st.session_state["adv_settings"]["halo_det"]["low_rad"],
            max_rad=st.session_state["adv_settings"]["halo_det"]["high_rad"],
            min_obj_size=st.session_state["adv_settings"]["halo_det"]["min_obj"],
            dil_disk=st.session_state["adv_settings"]["halo_det"]["disk"]
        )  

        # Assign halos to their spot and add the index of the spot to a list.
        halo_list=[]
        for s in sort_spots:
            s.assign_halo(halos)
            
            if s.halo>0:
                halo_list.append(s.row_name+str(s.col))

        # UI Selection of false-positive halos
        false_pos=st.multiselect("Remove false-positive Halos:",halo_list)

        for s in sort_spots:
            # If a false-positive was selected, remove the corresponding halo.
            if s.row_name+str(s.col) in false_pos:
                s.halo=np.nan

            # Scale halo radius to intensity of raw spot intensities
            if s.halo>0 and st.session_state["adv_settings"]["halo_det"]["tog_scale"]=="Scale to raw Data":
                s.int=s.halo/st.session_state["adv_settings"]["halo_det"]["scaling"]
            
            # Scale halo radius to intensity of normalized spot intensities
            elif s.halo>0 and st.session_state["adv_settings"]["halo_det"]["tog_scale"]=="Scale to normalized Data":
                s.norm_int=s.halo/st.session_state["adv_settings"]["halo_det"]["scaling"]

    # Tabs for all Results that are displayed
    tab1,tab2,tab3,tab4=st.tabs(["Image","Table","Heatmap", "Grid"])

    # saves spot-list in a session state
    st.session_state["current_img"]=sort_spots
    
    # Turns spotlist to df for visualization and download.
    df=msu.spot.create_df(sort_spots)
    
    figuredict={}
    # Displays image with main results.
    with tab1:
        st.markdown("## Detected Spots")
        fig_img,ax=plt.subplots()
        plots.plot_result(
            figure=fig_img,
            axs=ax,
            img=st.session_state["img"],
            df=df,
            g_prop=st.session_state["grid"],
            halo=st.session_state["halo_toggle"]
        )        
        st.pyplot(fig_img)
        figuredict["img_results"]=fig_img

    # Displays the Table containing all information of the spots
    with tab2:
        st.markdown("## Tabular Results")
        st.dataframe(df)

    # Displays a heatmap of spot-intensities
    with tab3:
        st.markdown("## Heatmap of Results")
        # Display Image and corresponding Heatmap
        fig_hm,ax=plt.subplots()
        plots.plot_heatmapv2(
            figure=fig_hm,
            axs=ax,
            df=df,
            conv_dict=row_conv_inv,
            norm_data=st.session_state["norm"],
            halo=st.session_state["halo_toggle"]
        )
        st.pyplot(fig_hm)
        figuredict["heatmap"]=fig_hm
    
    # Displays the detected grid.
    with tab4:
        col1,col2=st.columns(2)
        with col1:
            # Display the grid.
            fig_grid,ax=plt.subplots()
            plots.plot_grid(
                figure=fig_grid,
                axs=ax,
                img=st.session_state["img"],
                lines=hor_line+vert_line
            )
            for s in init_spots:
                ax.scatter(s.x,s.y, c="k", marker="x")
            st.pyplot(fig_grid)
            figuredict["detected_grid"]=fig_grid

            fig,ax=plt.subplots()
            ax.imshow(test["edge"])
            st.pyplot(fig)

        with col2: 
            st.markdown("## Detected Grid")
            mst.v_space(1)
            st.markdown("Please check whether the gridlines match the Spots!")
            st.markdown("A faulty grid leads to errors during spot detection and can influence the results negatively. The most frequent reason for faulty grids is a noisy background in the submitted image.")

    mst.v_space(1)

    c1,c2=st.columns(2)
    with c2:
        # Creates a Zipfile containing all plots as .svg files on this page and makes it ready for download.
        mst.download_figures(figuredict,"svg")
    
    with c1:
        # Creates a Zipfile containing all plots as .png files on this page and makes it ready for download.
        mst.download_figures(figuredict,"png")
