import streamlit as st
import pandas as pd

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
        "ctrl_rows":[],             # Row-Indexes of spots to be labeled as controls
        "ctrl_cols":[],             # Column_Indexes of spots to be labeled as controls
        "halo_toggle":True,         # Enables Halo detection if true.
        "false_pos":[],             # List of Spot-Indices with false-positive halos.
        }       

def set_analyze_True():
    # Callback function to set analyze state to true
    st.session_state["analyze"]=True

def page_setup():
    # Sets title and site icon
    st.set_page_config(
        page_title="Microspot Reader",
        page_icon=r"assets/Logo_notext.png",
        initial_sidebar_state="auto",
        menu_items=None
        )
    # Initializes all required session states
    for name,state in states.items():
        if name not in st.session_state:
            st.session_state[name]=state

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
    
def apply_datachange():
    # Applies all edits to image and merged data made by the user to the respective dataframe
    st.session_state["img_df"]=st.session_state["edit_img"]
    st.session_state["merge_df"]=st.session_state["edit_merge"]
    # Disables warning
    st.session_state["change_warning"]=False

def del_sessiondata():
    # Removes all selected rows from img_df and merge_df
    apply_datachange()
    st.session_state["img_df"]=st.session_state["img_df"].loc[st.session_state["edit_img"]["Select"]==False]
    st.session_state["merge_df"]=st.session_state["merge_df"].loc[st.session_state["edit_merge"]["Select"]==False]

    # Removes selected data from the dictionaries storing the actual lists
    st.session_state["img_data"]={st.session_state["img_df"].loc[idx,"id"]: st.session_state["img_data"][st.session_state["img_df"].loc[idx,"id"]] for idx in st.session_state["img_df"].index}
    st.session_state["merge_data"]={st.session_state["merge_df"].loc[idx,"id"]: st.session_state["merge_data"][st.session_state["merge_df"].loc[idx,"id"]] for idx in st.session_state["merge_df"].index}

def datachange_warning():
    # Callback function, activates the warning upon changes to the stored data.
    st.session_state["change_warning"]=True

# Displays data stored in the current session in the sidebar
def datainfo():
    with st.sidebar:
        st.markdown("### Data in current Session")
        
        if st.session_state["change_warning"]==True:
            # Warning if changes to stored data were not applied yet.
            st.warning("Changes have not been applied yet!")
        
        # Displays an editable table for the stored image data
        st.caption("Image-Data")
        st.session_state["edit_img"]=st.data_editor(st.session_state["img_df"],column_config={"Select":st.column_config.CheckboxColumn("Select",default=False),"id":None},use_container_width=True,hide_index=True,on_change=datachange_warning,key="1")

        # Displays an editable table for the stored merged data
        st.caption("Merged Data")
        st.session_state["edit_merge"]=st.data_editor(st.session_state["merge_df"],column_config={"Select":st.column_config.CheckboxColumn("Select",default=False),"id":None},use_container_width=True,hide_index=True,on_change=datachange_warning,key="2")

        # Buttons for deleting data and applying changes to data.
        col1,col2=st.columns(2)
        with col2:
            st.button("Delete Selection",on_click=del_sessiondata,use_container_width=True)
        with col1:        
            st.button("Apply Changes",on_click=apply_datachange,use_container_width=True,type="primary")

# Function to add vertical space between elements
def v_space(n, col=None):
    for _ in range(n):
        if col:
            col.write("")
        else:
            st.write("")

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

def set_falsepos():
    st.session_state["false_pos"]=st.session_state["false"]