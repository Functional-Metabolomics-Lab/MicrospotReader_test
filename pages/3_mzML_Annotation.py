import streamlit as st
import pandas as pd
import microspot_util as msu
import microspot_util.streamlit as mst
import microspot_util.plots as plots
import pyopenms as oms
from io import StringIO
from pyopenms.plotting import plot_spectrum,plot_chromatogram
import matplotlib.pyplot as plt
import numpy as np

mst.page_setup()

st.markdown("# Annotation of .mzML-Files")

mst.datainfo()

st.markdown("## Settings")

choose_input=st.selectbox("Upload of Merged Data:",["Use Selection in current Session","Upload Merged Data"])

c1,c2=st.columns(2)

with c1:
    spot_mz=st.number_input("Proxy m/z-value to be used for Annotation",min_value=1,value=1000,step=1)

with c2:
    intensity_scalingfactor=st.number_input("Scaling-Factor of Spot-Intensity",value=10e6)

with c1:
    mzml_upload=st.file_uploader("Upload .mzML File","mzML")

if choose_input=="Use Selection in current Session":
    data_id=st.session_state["merge_df"].loc[st.session_state["merge_df"]["Select"]==True,"id"]
    
    with c2:
        st.caption("Selected Dataset from Session:")
        st.dataframe(st.session_state["merge_df"].loc[st.session_state["merge_df"]["Select"]==True,"Name"],column_config={"Name":"Selected Dataset:"},use_container_width=True,hide_index=True)


    if len(data_id)==1:
        st.session_state["disable_mzml"]=False
        merged_data=st.session_state["merge_data"][st.session_state["merge_df"].loc[st.session_state["merge_df"]["Select"]==True,"id"].max()]
        st.dataframe(merged_data)
    
    else:
        st.session_state["disable_mzml"]=True
        st.warning("Please select __one__ merged dataset!")

else:
    with c2:
        merge_upload=st.file_uploader("Upload RT annotated Datatable",["csv","tsv"])
    
    if merge_upload != None:
        st.session_state["disable_mzml"]=False
        merged_data=pd.read_csv(merge_upload,index_col=0)
        st.dataframe(merged_data)
    
    else:
        st.session_state["disable_mzml"]=True

if mzml_upload == None:
    st.session_state["disable_mzml"]=True

_,c1,_,c2,_=st.columns([0.15,0.3,0.1,0.3,0.15])
with c1:
    if st.button("Annotate Data",disabled=st.session_state["disable_mzml"],type="primary",use_container_width=True):
        # Create new MS Experiment and load mzml file to it.
        mzml_string=StringIO(mzml_upload.getvalue().decode("utf-8")).read()
        exp=oms.MSExperiment()
        oms.MzMLFile().loadBuffer(mzml_string,exp)
        
        # Annotate the MS1 spectra in the MSExperiment class with interpolated spot-intensity values
        msu.annotate_mzml(exp,merged_data,spot_mz,intensity_scalingfactor)

        st.session_state["annot_mzml"]=exp
        
        mzml_output=oms.MzMLFile().store("output.mzML",st.session_state["annot_mzml"])
        st.session_state["mzml_download"]=False

with c2:
    if st.session_state["mzml_download"]==False:
        with open("output.mzML", "rb") as mzml_file:
            st.download_button(label="Download mzML File",data=mzml_file,mime="mzML",disabled=st.session_state["mzml_download"],use_container_width=True)

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