import io

import streamlit as st
import pandas as pd
import pyopenms as oms
from pyopenms.plotting import plot_chromatogram
import matplotlib.pyplot as plt

import microspot_util as msu
import microspot_util.streamlit as mst
import microspot_util.plots as plots

# Initialize session-states and add basic design elements.
mst.page_setup()
st.session_state["merge_state"]=False

with st.sidebar:
    # Displays data that has been saved in the current session in tabular format.
    mst.datainfo()

st.markdown("# Annotation of .mzML-Files")

st.markdown("## Settings")

# Choose to upload a .csv file or use data saved in the current session.
choose_input=st.selectbox("Upload of Merged Data:",["Use Selection in current Session","Upload Merged Data"])

c1,c2=st.columns(2)

with c1:
    # m/z value that is used as a proxy mass to save the spotintensity/bioactivity at at
    spot_mz=st.number_input("Proxy m/z-value to be used for Annotation",min_value=1,value=1000,step=1)

with c2:
    # Factor to scale the spot intensity with --> required for sequential analysis with mzmine
    intensity_scalingfactor=st.number_input("Scaling-Factor of Spot-Intensity",value=10e6)

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
    
    if merge_upload != None:
        st.session_state["disable_mzml"]=False
        merged_data=pd.read_csv(merge_upload,index_col=0)
        st.dataframe(merged_data)
    
    # Disable start button if nothing has been uploaded
    else:
        st.session_state["disable_mzml"]=True

# Disable startbutton if nothing has been uploaded
if mzml_upload == None:
    st.session_state["disable_mzml"]=True

_,c1,_,c2,_=st.columns([0.15,0.3,0.1,0.3,0.15])

with c1:
    # Initiate the Annotation if everything has been selected
    if st.button("Annotate Data",disabled=st.session_state["disable_mzml"],type="primary",use_container_width=True):
        # Create new MS Experiment and load mzml file to it.
        mzml_string=io.StringIO(mzml_upload.getvalue().decode("utf-8")).read()
        exp=oms.MSExperiment()
        oms.MzMLFile().loadBuffer(mzml_string,exp)
        
        # Annotate the MS1 spectra in the MSExperiment class with interpolated spot-intensity values
        msu.annotate_mzml(exp,merged_data,spot_mz,intensity_scalingfactor)

        # Save the experiment in a session state
        st.session_state["annot_mzml"]=exp
        
        st.session_state["mzml_download"]=False

with c2:
    # If Downloadbutton is enabled, show it and allow the download
    if st.session_state["mzml_download"]==False:
        # Store the mzml file on the server as an output, this is needed for the download button to work
        mst.download_mzml(st.session_state["annot_mzml"])

# Display the TIC-Chromatogram aswell as the bioactivity chromatogram, derived from the mzml file.
if st.session_state["annot_mzml"] is not None:
    t1,t2=st.tabs(["TIC-Chromatogram","Bioactivity Chromatogram"])
    with t2:
        fig,ax=plt.subplots()
        plots.plot_mzml_chromatogram(fig,ax,st.session_state["annot_mzml"],spot_mz)
        st.pyplot(fig)
    
    with t1:
        chrom=st.session_state["annot_mzml"].getChromatogram(0)
        fig,ax=plt.subplots()
        plot_chromatogram(chrom)
        ax.set(
        title="TIC-Chromatogramm",
        ylabel="TIC",
        xlabel="Retention Time [s]",
        )
        st.pyplot(fig)