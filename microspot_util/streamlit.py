import streamlit as st
import pandas as pd
import tempfile
import io
import zipfile
import pyopenms as oms
import os     

def init_sessionstate():
    """
    Initializes all required session states defined by a dictionary if they dont exist.
    """
    # All session states that need to be initialized:
    states={"analyze":False,            # State of image analysis -> False = not processed, True = processed
            "img_data":{},              # Dictionary that stores user specified spot-lists under a unique ID
            "init_analysis":True,       # Enables the "Start Analysis" button in image analysis if set to False
            "img":None,                 # Stores the current image as a np.array
            "grid":None,                # Stores the grid-parameters during image analysis
            "current_img":None,         # Stores the spot-list of the current image analysis
            "edit_img":None,            # df of the user edits to img_df, used to update the stored image data
            "change_warning":False,     # enables warning in sidebar after changing session data if changes were not saved
            "mergedata_loaded":True,    # enables the "Merge Data" button for data merging if set to false
            "merge_state":False,        # State of Data Merging -> False = not merged, True = merged
            "img_df":pd.DataFrame(columns=["Name","Select","id"]), # Dataframe for display of stored session data, used to index img_data dictionary
            "session_id":0,             # ID counter for stored data
            "edit_merge":None,          # df of the user edits to merge_df, used to update the stored merged data
            "merge_df":pd.DataFrame(columns=["Name","Select","id"]), # Dataframe for display of stored session data, used to index merged_data dictionary
            "merge_data":{},            # Dictionary that stores user specified merged spot-lists under a unique ID
            "current_merge":None,       # Stores the spot-list of the current data merge
            "disable_mzml":True,        # Enables the "Annotate Data" button for mzml annotation if set to False
            "mzml_download":True,       # Activates the Download-Button for current results.
            "annot_mzml":None,          # Stores the final annotated mzML file.
            "ctrl_rows":[],             # Rows not containing samples
            "ctrl_cols":[],             # Columns not containing sampls
            "ctrl_spots":[],            # Spot indices to be labeled as controls -> currently used for normalization
            "norm_toggle":True,         # Enables normalization of spots if True
            "halo_toggle":True,         # Enables Halo detection if true.
            "norm":False,               # Keeps track of wheter data has been normalized during image analysis.
            "adv_settings":{"init_det":{},"grid_det":{},"spot_misc":{},"halo_det":{}}, # initializes dictionary containing all values from the advanced settings
            "results":None,             # variable containing results from feature detection and annotation
            }

    for name,state in states.items():
        if name not in st.session_state:
            st.session_state[name]=state

def page_setup():
    # Sets title and site icon
    st.set_page_config(
        page_title="Microspot Reader",
        page_icon="assets/Icon.png",
        initial_sidebar_state="auto",
        menu_items={
            "About":"""
                    - Interested in contributing? Check out our [GitHub personal page](https://github.com/sknesiron).
                    - For more about our work, visit our [lab's GitHub page](https://github.com/Functional-Metabolomics-Lab).
                    - Follow us on [Twitter](https://twitter.com/Functional-Metabolomics-Lab) for the latest updates.

                    Made by Simon B. Knoblauch
                    """,
            "Report a Bug":"https://github.com/Functional-Metabolomics-Lab/MicrospotReader/issues/new"
        }
        )
    # Initializes all required session states
    init_sessionstate()

def v_space(n, col=None):
    """
    Utility function to add vertical space between elements
    """
    for _ in range(n):
        if col:
            col.markdown("")
        else:
            st.markdown("")

def datainfo():
    """
    Utility function to set up the data info panel for stored session data.
    """

    st.markdown("### Data in current Session")
    
    # Warning if changes to stored data were not applied yet.
    if st.session_state["change_warning"] is True:
        st.warning("Changes have not been applied yet!")
    
    # Displays an editable table for the stored image data
    st.caption("Image-Data")
    st.session_state["edit_img"]=st.data_editor(
        st.session_state["img_df"],
        column_config={
            "Select":st.column_config.CheckboxColumn("Select",default=False),
            "id":None
            },
        use_container_width=True,
        hide_index=True,
        on_change=datachange_warning,
        key="1"
        )

    # Displays an editable table for the stored merged data
    st.caption("Prepared Data")
    st.session_state["edit_merge"]=st.data_editor(
        st.session_state["merge_df"],
        column_config={
            "Select":st.column_config.CheckboxColumn("Select",default=False),
            "id":None
            },
        use_container_width=True,
        hide_index=True,
        on_change=datachange_warning,
        key="2"
        )

    # Buttons for deleting data and applying changes to data.
    col1,col2=st.columns(2)
    with col2:
        st.button("Delete Selection",on_click=del_sessiondata,use_container_width=True)

    with col1:        
        st.button("Apply Changes",on_click=apply_datachange,use_container_width=True,type="primary")

def datachange_warning():
    """
    Callback function, activates the warning upon changes to the stored data.
    """
    st.session_state["change_warning"]=True

def apply_datachange():
    """
    Callback function: Applies all edits to image and merged data made by the user to the respective dataframe.
    """
    st.session_state["img_df"]=st.session_state["edit_img"]
    st.session_state["merge_df"]=st.session_state["edit_merge"]
    # Disables warning
    st.session_state["change_warning"]=False
    reset_merge()

def del_sessiondata():
    """
    Callback function: Removes all selected rows from img_df and merge_df
    """

    apply_datachange()
    
    st.session_state["img_df"]=st.session_state["img_df"].loc[st.session_state["edit_img"]["Select"]==False]

    st.session_state["merge_df"]=st.session_state["merge_df"].loc[st.session_state["edit_merge"]["Select"]==False]

    # Removes selected data from the dictionaries storing the actual lists
    st.session_state["img_data"]={st.session_state["img_df"].loc[idx,"id"]: st.session_state["img_data"][st.session_state["img_df"].loc[idx,"id"]] for idx in st.session_state["img_df"].index}
    st.session_state["merge_data"]={st.session_state["merge_df"].loc[idx,"id"]: st.session_state["merge_data"][st.session_state["merge_df"].loc[idx,"id"]] for idx in st.session_state["merge_df"].index}
    reset_merge()

def add_imgdata(data_name):
    # Stores spot-list of current image in img_data with a unique session ID
    st.session_state["img_data"][st.session_state["session_id"]]=st.session_state["current_img"]

    # Stores user-given name and id in img_df dataframe. displayed in sidebar
    add_data=pd.Series({"Name":data_name,"Select":False,"id":st.session_state["session_id"]})
    st.session_state["img_df"]=pd.concat([st.session_state["img_df"],add_data.to_frame().T],ignore_index=True)

    # Increase ID counter
    st.session_state["session_id"]+=1

def add_mergedata(data_name):
    # Stores merged data in merge_date with a unique session ID
    st.session_state["merge_data"][st.session_state["session_id"]]=st.session_state["current_merge"]
    
    # Stores user-given name and id in merge_df dataframe. displayed in sidebar
    add_data=pd.Series({"Name":data_name,"Select":False,"id":st.session_state["session_id"]})
    st.session_state["merge_df"]=pd.concat([st.session_state["merge_df"],add_data.to_frame().T],ignore_index=True)

    # Increase ID counter
    st.session_state["session_id"]+=1

@st.cache_resource
def convert_df(df):
    # function to turn df to a .csv, prepares table for download
    return df.to_csv().encode("utf-8")

def mergedata_loaded():
    # Callback function, enables the "Merge Data" button.
    st.session_state["mergedata_loaded"]=False

def merge_settings():
    # Callback function to start the merging workflow
    st.session_state["merge_state"]=True

def reset_merge():
    # Callback function to reset the merge state
    st.session_state["merge_state"]=False

def set_analyze_True():
    # Callback function to set analyze state to true
    st.session_state["analyze"]=True

def temp_figurefiles(figure_dict,suffix,directory):
    pathlist=[]
    for figname, figure in figure_dict.items():
        figpath=os.path.join(directory,f"{figname}.{suffix}")
        figure.savefig(figpath,format=suffix,dpi=300)
        pathlist.append(figpath)
    return pathlist

@st.cache_resource
def temp_zipfile(paths):
    with tempfile.NamedTemporaryFile(delete=False) as tempzip:
        with zipfile.ZipFile(tempzip.name,"w") as temp_zip:
            for figurepath in paths:
                temp_zip.write(figurepath,arcname=os.path.basename(figurepath))
    return tempzip

def download_figures(figure_dict:dict,suffix:str="svg") -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        paths=temp_figurefiles(figure_dict,suffix,tempdir)

        tempzip=temp_zipfile(paths)
            
        with open(tempzip.name,"rb") as zip_download:
            st.download_button(
                label=f"Download Plots as .{suffix}",
                data=io.BytesIO(zip_download.read()),
                mime=".zip",
                file_name="plots_img-analysis.zip",
                use_container_width=True
            )

def download_gnpsmgf(consensus_map:oms.ConsensusMap,mzmlfilename:str,exp:oms.MSExperiment):
    filtered_map=oms.ConsensusMap(consensus_map)
    filtered_map.clear(False)
    for feature in consensus_map:
        if feature.getPeptideIdentifications():
            filtered_map.push_back(feature)
    
    with tempfile.NamedTemporaryFile() as temp:
        oms.ConsensusXMLFile().store(temp.name, filtered_map)

        with tempfile.TemporaryDirectory() as tempdir:
            mgf_name=os.path.join(tempdir,"MS2data.mgf")
            quant_name=os.path.join(tempdir,"FeatureQuantificationTable.txt")
            mzml_name=os.path.join(tempdir,mzmlfilename)
            
            oms.MzMLFile().store(mzml_name, exp)

            oms.GNPSMGFFile().store(
                oms.String(temp.name),
                [mzml_name.encode()],
                oms.String(mgf_name)
            )
            oms.GNPSQuantificationFile().store(consensus_map, quant_name)
            
            tempzip=temp_zipfile([mzml_name,mgf_name,quant_name])

    with open(tempzip.name,"rb") as zip_download:
        st.download_button(
            label=f"Download Files for FBMN",
            data=io.BytesIO(zip_download.read()),
            mime=".zip",
            file_name="fbmn_files.zip",
            use_container_width=True,
            type="primary"
        )

def download_mzml(exp):
    with tempfile.NamedTemporaryFile(suffix=".mzML", delete=False) as mzml_file:
        oms.MzMLFile().store(mzml_file.name, exp)

    with open(mzml_file.name,"rb") as mzml_file:
        st.download_button(
            label="Download .mzML File",
            data=io.BytesIO(mzml_file.read()),
            mime=".mzML",
            file_name="annotated_file.mzML",
            use_container_width=True
            )