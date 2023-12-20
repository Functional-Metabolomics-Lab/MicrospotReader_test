import io

import streamlit as st
import pyopenms as oms
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import microspot_reader as msu
import microspot_reader.streamlit as mst
import microspot_reader.plots as plots

# Dictionaries to convert Row-Letters into Row-Numbers and vice versa (required for heatmap)
row_conv={"abcdefghijklmnopqrstuvwxyz"[i-1]: i for i in range(1,27)}
row_conv_inv={v:k for k,v in row_conv.items()}

# Initialize session-states and add basic design elements.
mst.page_setup()
st.session_state["merge_state"]=False

with st.sidebar:
    # Displays data that has been saved in the current session in tabular format.
    mst.datainfo()

st.markdown("# Feature finding and annotation with activity data")

# # Choose to upload a .csv file or use data saved in the current session.
# choose_input=st.selectbox("Upload of Merged Data:",["Use Selection in current Session","Upload Merged Data"])

# choose_mzml=st.selectbox("Upload of .mzML-File:",["Upload .mzML File","Example .mzML File"])

c1,c2=st.columns(2)

with c2:
    choose_mzml=st.selectbox("Upload of .mzML-File:",["Upload .mzML File","Example .mzML File"])
    # File upload for .mzml to be annotated
    if choose_mzml=="Upload .mzML File":
        mzml_upload=st.file_uploader("Upload .mzML File","mzML")
    else:
        mzml_upload="example_files/example_mzml.mzML"
        mst.v_space(3)
        st.info("**Example .mzML File**")

with c1:
    # Choose to upload a .csv file or use data saved in the current session.
    choose_input=st.selectbox("Upload of Merged Data:",["Use Selection in current Session","Upload Merged Data"])

if choose_input=="Use Selection in current Session":
    # choose data from the current session
    data_id=st.session_state["merge_df"].loc[st.session_state["merge_df"]["Select"]==True,"id"]
    
    with c1:
        # Display the chosen data by name
        st.caption("Selected Dataset from Session:")
        st.dataframe(st.session_state["merge_df"].loc[st.session_state["merge_df"]["Select"]==True,"Name"],column_config={"Name":"Selected Dataset:"},use_container_width=True,hide_index=True)

    if len(data_id)==1:
        # Only continue with analysis if a single dataset has been chosen
        st.session_state["disable_mzml"]=False
        # Load the dataframe into merged_data variable and then display it
        merged_data=st.session_state["merge_data"][st.session_state["merge_df"].loc[st.session_state["merge_df"]["Select"]==True,"id"].max()]
        st.dataframe(merged_data)
    
    else:
        # Disable the "Start Annotation" button if more or less than 1 data set have been selected
        st.session_state["disable_mzml"]=True
        st.warning("Please select __one__ merged dataset!")

else:
    with c1:
        # File uploader for .csv containing merged data
        merge_upload=st.file_uploader("Upload RT annotated Datatable",["csv","tsv"])
    
    if merge_upload is not None:
        st.session_state["disable_mzml"]=False
        merged_data=pd.read_csv(merge_upload,index_col=0)
        st.dataframe(merged_data)
    
    # Disable start button if nothing has been uploaded
    else:
        st.session_state["disable_mzml"]=True

if mzml_upload is None:
    st.session_state["disable_mzml"]=True

with st.form("Settings"):

    c1,c2=st.columns(2)

    with c1:
        mass_error=st.number_input(
            "Mass Error for feature detection*[in ppm]*:",
            value=10.0
        )
        
        min_fwhm=st.number_input(
            "Min. fwhm of peaks during feature detection *[in s]*:",
            value=1.0
        )

        rt_tolerance=st.number_input(
            "Retention Time tolerance *[in s]*",
            value=1
        )

        threshold_method=st.selectbox(
            "Method for activity-peak threshold determination:",
            ["Automatic","Manual"],
            index=0
        )

        st.markdown("####")

        # Initiaize analysis:
        analysis=st.form_submit_button(
            "Start Feature Detection and Annotation",
            disabled=st.session_state["disable_mzml"],
            type="primary",
            use_container_width=True
        )

    with c2:
        noise_threshold=st.number_input(
            "Noise Threshold for feature detection:",
            value=1e5
        )

        max_fwhm=st.number_input(
            "Max. fwhm of peaks during feature detection *[in s]*:",
            value=60.0
        )
        
        rt_offset=st.number_input(
            "Retention Time offset *[in s]*",
            value=4
        )

        baseline_conv=st.number_input(
            "Peak-Threshold criteria (activity data):",
            value=0.02,
            format="%f"
        )

        value_col=st.selectbox(
            "Column containing activity data:",
            ["norm_intensity","spot_intensity","smoothed_int"],
            index=2
        )

    # Initiate the analysis if everything has been selected
    if analysis:

        if choose_mzml=="Upload .mzML File":
            # Create new MS Experiment and load mzml file to it.
            mzml_string=io.StringIO(mzml_upload.getvalue().decode("utf-8")).read()
            exp=oms.MSExperiment()
            oms.MzMLFile().loadBuffer(mzml_string,exp)
            filename=mzml_upload.name

        else:
            exp=oms.MSExperiment()
            oms.MzMLFile().load(mzml_upload,exp)
            filename="example_mzml.mzML"

        # Feature Detection and creation of a feature table from the mzml file.
        fm=msu.feature_finding(
            exp=exp,
            filename=filename,
            mass_error=mass_error,
            noise_threshold=noise_threshold,
            min_fwhm=float(min_fwhm),
            max_fwhm=float(max_fwhm),
        )

        # Maps ms2 spectra to features in the created featuremap for export of .mgf file 
        msu.ms2_mapping(
            exp=exp,
            fm=fm
        )

        # Adduct detection in fm, might be unreliable
        ft,consensus_map=msu.adduct_detector(
            fm=fm,
            adduct_list=[
                b'H:+:0.4', 
                b'Na:+:0.2', 
                b'NH4:+:0.2', 
                b'H3O1:+:0.1', 
                b'CH2O2:+:0.1', 
                b"H-2O-1:0:0.2"
            ]
        )

        # Generation of xics for all features in the featuretable ft.
        xic_dict,specta=msu.xic_generator(
            exp=exp,
            ft=ft
        )

    # Peak-detection in the activity chromatogram.
        merged_data.sort_values("RT",inplace=True)
        # determine peak-threshold based on user input
        if threshold_method == "Manual":
            pk_threshold=baseline_conv
        
        else:
            # small algorithm that tries to estimate noise by removing outliers (peaks) from the dataset
            stdev,mn=msu.baseline_noise(
                merged_data[value_col],
                baseline_conv
            )
            pk_threshold=mn+3*stdev

        # peak detection in 2d -> essentially a heatmap of the spot-data is used to determine the location of peaks
        # This is due to the fact that spots showing activity lead to increased intensites in a radius around the spot itself, if peak detection would be performed in the 1d chromatogram, these proximal regions to the active spot might also be detected as peaks even though they are not caused by active compounds. peakdetection in 2d circumvents this problem by only allowing spots to be peak-maxima when no surrounding spots show higher intensity.
        merged_data,aft,pk=msu.img_peak_detection(
            df=merged_data,
            datacolumn_name=value_col,
            threshold=pk_threshold
        )

        # feature table from ms feature detection is annotated with activity data through matching of retention time and ranking of the pearson correlation coefficient between the peaks. the value in the correlated activity column is the AUC of the activity peak. 
        msu.activity_annotation_features(
            ft=ft,
            aft=aft,
            rt_tolerance=rt_tolerance,
            rt_offset=rt_offset,
            act_df=merged_data,
            xic_dict=xic_dict,
            ydata_name=value_col
        )

        # saves the results in a dictionary for display so that this alogrithm does not have to run everytime the user interacts with the web app
        st.session_state["results"]={
            "featuretable":ft,
            "activitytable":aft,
            "ft_peaks":{f"peak{pk}":ft.loc[ft[f"corr_activity_peak{pk}"]>0].copy() for pk in aft.index},
            "xics":xic_dict,
            "spot_df":merged_data,
            "val_col":value_col,
            "baselineconv":pk_threshold,
            "consensus_map":consensus_map,
            "mzml_name":filename,
            "MSExperiment":exp,
            "peaks":pk
        }
        
if st.session_state["results"] is not None:

    st.markdown("## Results")

    c=st.container()

    t1,t2,t3=st.tabs(["Activity peak data",'Significant active features','Complete feature table'])

    with t3:
        # Displays the full feature table
        st.dataframe(st.session_state["results"]["featuretable"])

    with t1:
        # Displays key information of every peak
        st.dataframe(st.session_state["results"]["activitytable"], use_container_width=True)

        # Plot the 1d activity chromatogram, some apparent peaks will not be detected as these are artifacts from imaging.
        # Sometimes spots that show activity, influence their neighbouring spots aswell, leading to multiple (if weaker) peaks for the same activity at different time points. this is why peak detection of activity peaks is done in 2d.
        fig,ax=plt.subplots()
        plots.plot_activity_chromatogram(
            figure=fig,
            axs=ax,
            spot_df=st.session_state["results"]["spot_df"],
            peak_df=st.session_state["results"]["activitytable"],
            ydata_name=st.session_state["results"]["val_col"],
            peak_threshold=st.session_state["results"]["baselineconv"]
        )
        st.pyplot(fig)

        fig,ax=plt.subplots()
        # Plotting heatmap to showcase the detected peaks in 2d, as it might be easier to understand why only those have been detected
        plots.plot_heatmapv2(
            fig,
            ax,
            st.session_state["results"]["spot_df"],
            row_conv_inv,
            value_col=st.session_state["results"]["val_col"],
            colorbar_name="Spot Intensity [a.u.]",
            halo=any(st.session_state["results"]["spot_df"]["halo"]>0)
        )
        # Add location of detected peaks to heatmap
        ax.scatter(
            st.session_state["results"]["spot_df"].loc[st.session_state["results"]["activitytable"]["peak_idx"],"column"],
            -st.session_state["results"]["spot_df"].loc[st.session_state["results"]["activitytable"]["peak_idx"],"row"],
            c="r",
            marker="D",
        )
        # Write name of peak to corresponding spot
        for i in st.session_state["results"]["activitytable"].index:
            ax.text(
                st.session_state["results"]["spot_df"].loc[st.session_state["results"]["activitytable"].loc[i,"peak_idx"],"column"]+0.2,
                -st.session_state["results"]["spot_df"].loc[st.session_state["results"]["activitytable"].loc[i,"peak_idx"],"row"]+0.2,
                f"peak{i}",
                size=8,
                c="r",
                path_effects=[pe.withStroke(linewidth=1, foreground="white")]
            )
        fig.tight_layout()
        st.pyplot(fig)

    with t2:
        # Threshold of pearson correlation coefficient, features with lower correlation to the activity peak will not be seen as significant
        threshold=st.number_input(
            "Correlation Threshold (Pearson Correlation):",
            max_value=1.0,
            min_value=0.0,
            value=0.8
        )
        # List of peaks to select for peak shape overlay, to be appended with all detected peaks
        namelist=["Don't Display"]
        # display a dataframe for all detected activity peaks containing information on all features with significant correlation
        for name,pft in st.session_state["results"]["ft_peaks"].items():
            st.markdown(f"**Significant features for `{name}`**")
            st.dataframe(pft.loc[pft[f"pearson_corr_{name}"]>=threshold].sort_values(f"pearson_corr_{name}",ascending=False))
            namelist.append(name)
            mst.v_space(2)
        
        # Select peak for which to display the alignment of activity and feature peak
        dip_featurealignment=st.radio(
            "Display alignment of activity peak with significant features:",
            namelist,
            index=0
        )

        # Display a plot for each significant feature showing the alignment of the selected activity peak to it
        if dip_featurealignment!="Don't Display":
            
            # select only the significant features and sort them by correlation coefficient
            ft=st.session_state['results']['ft_peaks'][dip_featurealignment].loc[st.session_state['results']['ft_peaks'][dip_featurealignment][f"pearson_corr_{dip_featurealignment}"]>threshold].sort_values(f"pearson_corr_{dip_featurealignment}",ascending=False)

            # get the index of the activity peak
            peakidx=int(dip_featurealignment.removeprefix("peak"))
            
            for i in ft.index:
                # get the xic of the current feature
                df=st.session_state["results"]["xics"][i].copy()

                # plot the xic of the current feature, with the feature peak maximum moved to 0
                fig,ax=plt.subplots()
                ax.plot(
                    df.rt-ft.loc[i,"RT"],
                    df.int,
                    c="darkviolet"
                    )
                
                ax2=ax.twinx()

                # get the values for the retention time of the activity peak witht the peak maximum moved to 0
                activity_rt=st.session_state["results"]["spot_df"].loc[st.session_state["results"]["activitytable"].loc[peakidx,"start_idx"]:st.session_state["results"]["activitytable"].loc[peakidx,"end_idx"],"RT"]-st.session_state["results"]["activitytable"].loc[peakidx,"RT"]

                # get the intensity values of the activity peak
                activity_int=st.session_state["results"]["spot_df"].loc[st.session_state["results"]["activitytable"].loc[peakidx,"start_idx"]:st.session_state["results"]["activitytable"].loc[peakidx,"end_idx"],"smoothed_int"]
                
                # plot the activity peak
                ax2.plot(
                    activity_rt,
                    activity_int,
                    c="lime"
                )
                
                # Settings
                ax.set(
                    ylabel="Intensity MS-signal [a.u.]",
                    xlabel="ΔRT from Peak-maximum [s]",
                    xlim=[activity_rt.min()-5,activity_rt.max()+5],
                    ylim=[df.loc[(df.rt>ft.loc[i,"RTstart"]) & (df.rt<ft.loc[i,"RTend"]),"int"].min(), df.loc[(df.rt>ft.loc[i,"RTstart"]) & (df.rt<ft.loc[i,"RTend"]),"int"].max()]
                )

                ax2.set(
                    ylabel="Intensity activity-signal [a.u.]",
                    ylim=[activity_int.min(),activity_int.max()]
                )
                fig.legend(
                    ["Feature-peak","Activity-peak"],
                    loc="upper right",
                    bbox_to_anchor=(0.9,0.94),
                    title=f"m/z: {ft.loc[i,'mz']:.4f} \nRT: {ft.loc[i,'RT']:.1f}s \nCorr. coeff: {ft.loc[i,f'pearson_corr_{dip_featurealignment}']:.4f}"
                )
                fig.tight_layout()
                st.pyplot(fig)

    with c:
        # Download of a zip file containing 
        mst.download_gnpsmgf(
            st.session_state["results"]["consensus_map"],
            st.session_state["results"]["mzml_name"],
            st.session_state["results"]["MSExperiment"]
            )