import streamlit as st
import pyopenms as oms
import pandas as pd
import microspot_util as msu
import microspot_util.streamlit as mst
import microspot_util.plots as plots
import io
import matplotlib.pyplot as plt

# Initialize session-states and add basic design elements.
mst.page_setup()
st.session_state["merge_state"]=False

with st.sidebar:
    # Displays data that has been saved in the current session in tabular format.
    mst.datainfo()

st.markdown("# Feature finding and annotation with activity data")

# Choose to upload a .csv file or use data saved in the current session.
choose_input=st.selectbox("Upload of Merged Data:",["Use Selection in current Session","Upload Merged Data"])

c1,c2=st.columns(2)

with c1:
    # File upload for .mzml to be annotated
    mzml_upload=st.file_uploader("Upload .mzML File","mzML")

if choose_input=="Use Selection in current Session":
    # choose data from the current session
    data_id=st.session_state["merge_df"].loc[st.session_state["merge_df"]["Select"]==True,"id"]
    
    with c2:
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
    with c2:
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
            "Mass Error *[in ppm]*:",
            value=10.0
        )
        
        min_fwhm=st.number_input(
            "Min. fwhm of peaks during feature detection *[in s]*:",
            value=1.0
        )

        rt_tolerance=st.number_input(
            "Retention Time tolerance *[in s]*",
            value=5
        )

        baseline_conv=st.number_input(
            "Convergence criteria for baseline determination:",
            value=0.02
        )

    with c2:
        noise_threshold=st.number_input(
            "Noise Threshold:",
            value=10e5
        )

        max_fwhm=st.number_input(
            "Max. fwhm of peaks during feature detection *[in s]*:",
            value=60.0
        )

        value_col=st.selectbox(
            "Column containing activity data:",
            ["norm_intensity","spot_intensity"],
            index=0
        )

        st.markdown("#### ")

        # Initiaize analysis:
        analysis=st.form_submit_button(
            "Start Feature Detection and Annotation",
            disabled=st.session_state["disable_mzml"],
            type="primary",
            use_container_width=True
        )

    # Initiate the Annotation if everything has been selected
    if analysis:
        # Create new MS Experiment and load mzml file to it.
        mzml_string=io.StringIO(mzml_upload.getvalue().decode("utf-8")).read()
        exp=oms.MSExperiment()
        oms.MzMLFile().loadBuffer(mzml_string,exp)
        
        # Feature Detection and creation of a feature table from the mzml file.
        ft=msu.feature_finding(
            exp=exp,
            mass_error=mass_error,
            noise_threshold=noise_threshold,
            min_fwhm=float(min_fwhm),
            max_fwhm=float(max_fwhm),
        )

        # Peak-detection in the activity chromatogram.
        merged_data.sort_values("RT",inplace=True)
        aft=msu.peak_detection(
            df=merged_data,
            baseline_convergence=baseline_conv,
            rel_height=0.95,
            datacolumn_name=value_col,
        )

        msu.activity_annotation_features(
            ft=ft,
            aft=aft,
            rt_tolerance=rt_tolerance,
        )
        
        st.session_state["results"]={"featuretable":ft,"activitytable":aft,"spot_df":merged_data,"val_col":value_col,"baselineconv":baseline_conv}
        
if st.session_state["results"] is not None:

    t1,t2,t3=st.tabs(["Annotated feature table","Activity chromatogram","Peak data"])

    with t1:
        st.dataframe(st.session_state["results"]["featuretable"])

    with t2:
        fig,ax=plt.subplots()
        plots.plot_activity_chromatogram(
            figure=fig,
            axs=ax,
            spot_df=st.session_state["results"]["spot_df"],
            peak_df=st.session_state["results"]["activitytable"],
            ydata_name=st.session_state["results"]["val_col"],
            baseline_acceptance=st.session_state["results"]["baselineconv"]
        )
        st.pyplot(fig)
    
    with t3:
        st.dataframe(st.session_state["results"]["activitytable"])