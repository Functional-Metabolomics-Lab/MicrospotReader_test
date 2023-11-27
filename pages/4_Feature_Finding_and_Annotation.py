import streamlit as st
import pyopenms as oms
import scipy.signal as signal
import pandas as pd
import numpy as np
import microspot_util.streamlit as mst
import io

# Initialize session-states and add basic design elements.
mst.page_setup()
st.session_state["merge_state"]=False

with st.sidebar:
    # Displays data that has been saved in the current session in tabular format.
    mst.datainfo()

st.markdown("# Feature finding and annotation with activity data")

mzml_upload=st.file_uploader("Upload .mzML File","mzML")

if mzml_upload:
    # Create new MS Experiment and load mzml file to it.
    mzml_string=io.StringIO(mzml_upload.getvalue().decode("utf-8")).read()
    exp=oms.MSExperiment()
    oms.MzMLFile().loadBuffer(mzml_string,exp)