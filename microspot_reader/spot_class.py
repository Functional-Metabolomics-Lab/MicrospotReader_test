# Importing dependencies
import pandas as pd
import skimage
import numpy as np
import streamlit as st


class spot:
    """
    ## Description

    Class containing information on spots detected during spot-detection

    ## __init__ Input:

    |Parameter|Type|Description|
    |---|---|---|
    |x|float|x-coordinate of the spot|
    |y|float|y-coordinate of the spot|
    |rad|int|radius of the spot|
    |halo_rad|int|radius of the halo of the spot|
    |int|float|Average pixel-intensity of the spot|
    |note|str|Arbitrary String|
    |row|int|Row-Index of spot in grid|
    |col|int|Column-Index of spot in grid|
    |row_name|str|Name of Row|
    |rt|float|Retention Time of Spot in s|

    ## Class Attributes

    |Name|Description|
    |---|---|
    |x|x-coordinate of the spot|
    |y|y-coordinate of the spot|
    |rad|radius of the spot|
    |halo|radius of the halo of the spot|
    |int|Average pixel-intensity of the spot|
    |note|Arbitrary String|
    |row|Row-Index of spot in grid|
    |col|Column-Index of spot in grid|
    |row_name|Name of Row|
    |rt|Retention Time of Spot in s|
    """
    def __init__(self,x:float,y:float,rad:int=0,halo_rad=np.nan,int=np.nan,note="Initial Detection",row:int=np.nan,col:int=np.nan,row_name:str=np.nan,rt=np.nan,sample_type:str="Sample",norm_int:float=np.nan) -> None:
        self.x=x
        self.y=y
        self.rad=rad
        self.halo=halo_rad
        self.int=int
        self.note=note
        self.row=row
        self.col=col
        self.row_name=row_name
        self.rt=rt
        self.type=sample_type
        self.norm_int=norm_int
    
    def assign_halo(self,halo_list:list,dist_thresh:float=14.73402725) -> None:
        """
        ## Description

        Checks a list of halos for a match and assigns it to the spot.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |halo_list|int|List of halo-objects|
        |dist_thresh|float|Distance threshold for acceptance of halo|
        """
        for h in halo_list:
            if np.linalg.norm(np.array((h.x,h.y))-np.array((self.x,self.y)))<dist_thresh:
                self.halo=h.rad
            
    def get_intensity(self,img:np.array,rad:int=None) -> None:
        """
        ## Description

        Determines the average pixel-intensity of a spot in an image.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |img|np.array|Image to extract spot-intensity from|
        |rad|int|radius to be used for intensity determination, if None or 0: use the radius determined by spot-detection|
        
        ## Output

        Avgerage intensity of pixels in spot.
        """
        if rad == None or rad == 0:
            radius=self.rad
        else:
            radius=rad

        try:
            # Indices of all pixels part of the current spot
            rr,cc=skimage.draw.disk((self.y,self.x),radius)
            # Mean intensity of all pixels within the spot
            self.int=img[rr,cc].sum()/len(rr)
            return self.int
    
        except:
            print(f"Spot at Coordinates ({self.x}, {self.y}) could not be evaluated: (Partly) Out of Bounds.")
    
    def append_df(self,spot_df:pd.DataFrame) -> pd.DataFrame:
        """
        ## Description

        Adds current spot to an existing spot-DataFrame

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_df|DataFrame|DataFrame that the current spot should be added to|

        ## Output

        Updated DataFrame
        """
        new_df=pd.concat([spot_df,pd.Series({"row":self.row,
                                             "row_name":self.row_name,
                                             "column":self.col,
                                             "type":self.type,
                                             "x_coord":self.x,
                                             "y_coord":self.y,
                                             "radius":self.rad,
                                             "halo":self.halo,
                                             "spot_intensity":self.int,
                                             "norm_intensity":self.norm_int,
                                             "note":self.note,
                                             "RT":self.rt,
                                             }).to_frame().T],ignore_index=True)
        return new_df

    def draw_spot(self,image:np.array,value:float=1,radius:int=None) -> np.array:
        """
        ## Description

        Inserts the spot into an image with the given value and radius.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |image|Array|Image that the spot will be inseted into.|
        |value|float|Numeric value of the pixels of the spot.|
        |radius|int|Radius of the spot to be drawn if none is given, the radius attribute is called.|

        ## Output

        Array of Image with the spot inserted.
        """
        
        # Get the radius of the spot using its rad attribute if no radius is given.
        if radius==None: radius=self.rad
        
        # Get the indices of the spot
        rr,cc=skimage.draw.disk((self.y,self.x),radius)

        # Draw the spot into the image, if the spot is out of bounds return Message.
        try:
            image[rr,cc]=value
        except:
            print(f"Spot at Coordinates ({self.x}, {self.y}) could not be drawn: Out of Bounds.")
        return image

    @staticmethod
    @st.cache_data
    def detect(gray_img:np.array,spot_nr:int,canny_sig:int=10,canny_lowthresh:float=0.001,canny_highthresh:float=0.001,hough_minx:int=70,hough_miny:int=70,hough_thresh:float=0.3,small_rad:int=20,large_rad:int=30,troubleshoot:bool=False) -> list:
        """
        ## Description

        Crude detection of spots in a grayscale image.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |gray_img|np.array|Grayscale np.array of image to be analyzed|
        |spot_nr|int|Maximum number of spots to be detected in the image|
        |canny_sig|int|Sigma value for gaussian blur during canny edge detection|
        |canny_lowthresh|float|Low threshold for canny edge detection|
        |canny_highthres|float|High threshold for canny edge detection|
        |hough_minx|int|Miniumum distance in x direction between 2 peaks during circle detection using hough transform|
        |hough_miny|int|Miniumum distance in y direction between 2 peaks during circle detection using hough transform|
        |hough_thresh|int|threshold of peak-intensity during circle detection using hough transform as fraction of maximum value|
        |small_rad|int|Smallest tested radius|
        |large_rad|int|Largest tested radius|

        ## Output

        List of detected spots as spot-objects
        """

        histeq_img=skimage.filters.rank.equalize(skimage.util.img_as_ubyte(gray_img),skimage.morphology.disk(50))
        edges=skimage.feature.canny(
        image=histeq_img,
        sigma=canny_sig,
        low_threshold=canny_lowthresh,
        high_threshold=canny_highthresh
        )

        # Range of Radii that are tested during inital spotdetection.
        tested_radii=np.arange(small_rad,large_rad+1)

        # Hough transform for a circle of the edge-image and peak detection to find circles in earlier defined range of radii.
        spot_hough=skimage.transform.hough_circle(edges,tested_radii)
        accums,spot_x,spot_y,spot_rad=skimage.transform.hough_circle_peaks(
            hspaces=spot_hough,
            radii=tested_radii,
            total_num_peaks=spot_nr,
            min_xdistance=hough_minx,
            min_ydistance=hough_miny,
            threshold=hough_thresh*spot_hough.max()
            )
        
        spotlist=[spot(x,y,rad) for x,y,rad in zip(spot_x,spot_y,spot_rad)]
        
        if troubleshoot is False:
            return spotlist
        else:
            return spotlist, {"edge":edges,"hough":spot_hough}
    
    @staticmethod
    def annotate_RT(spot_list:list,start:float,end:float)->list:
        """
        ## Description

        Annotates a list of spot-objects with Retention-Times.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be sorted|
        |start|float|Timepoint at which spotting was started in seconds|
        |end|float|Timepoint at which spotting was stopped|

        ## Output

        RT-annotated list of spot-objects
        """
        for s,rt in zip(spot_list,np.linspace(start,end,num=len(spot_list))):
            s.rt=rt
    
    @staticmethod
    def sort_list(spot_list:list,serpentine:bool=False,inplace:bool=False)->list:
        """
        ## Description

        Sorts a list of spot-objects by rows and columns.
        If serpentine is set to true even rows are sorted in a descending order.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be sorted|
        |serpentine|bool|True if even rows should be sorted in a descending order, False if even rows should be sorted in an ascending order|
        |inplace|bool|True if list should be sorted in place|

        ## Output

        Sorted list of spot-objects
        """

        if inplace==True:
            if serpentine:
                spot_list.sort(key=lambda x: x.row*1000+(x.row%2)*x.col-((x.row+1)%2)*x.col)
            else:
                spot_list.sort(key=lambda x: x.row*1000+x.col)
        else:
            if serpentine:
                sort=sorted(spot_list,key=lambda x: x.row*1000+(x.row%2)*x.col-((x.row+1)%2)*x.col)
            else:
                sort=sorted(spot_list,key=lambda x: x.row*1000+x.col)
            
            return sort

    
    @staticmethod
    def create_df(spot_list:list) -> pd.DataFrame:
        """
        ## Description

        Creates a DataFrame from a list of spots.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be turned into a DataFrame|

        ## Output

        DataFrame of spot-list.
        """

        spot_df=pd.DataFrame({"row":[i_spot.row for i_spot in spot_list],
                              "row_name":[i_spot.row_name for i_spot in spot_list],
                              "column":[i_spot.col for i_spot in spot_list],
                              "type":[i_spot.type for i_spot in spot_list],
                              "x_coord":[i_spot.x for i_spot in spot_list],
                              "y_coord":[i_spot.y for i_spot in spot_list],
                              "radius":[i_spot.rad for i_spot in spot_list],
                              "halo":[i_spot.halo for i_spot in spot_list],
                              "spot_intensity":[i_spot.int for i_spot in spot_list],
                              "norm_intensity":[i_spot.norm_int for i_spot in spot_list],
                              "note":[i_spot.note for i_spot in spot_list],
                              "RT":[i_spot.rt for i_spot in spot_list]})
        return spot_df
    
    @staticmethod
    def df_to_list(df:pd.DataFrame)->list:
        """
        ## Description

        Creates a list of spot-objects from a DataFrame created by the spot.create_df method.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |df|DataFrame|DataFrame created by the spot.create_df method|

        ## Output
        List of spot-objects
        """
        
        spot_list=[]
        
        for idx in df.index:
            spot_list.append(spot(x=df.loc[idx,"x_coord"],
                                  y=df.loc[idx,"y_coord"],
                                  rad=df.loc[idx,"radius"],
                                  halo_rad=df.loc[idx,"halo"],
                                  int=df.loc[idx,"spot_intensity"],
                                  note=df.loc[idx,"note"],
                                  row=df.loc[idx,"row"],
                                  col=df.loc[idx,"column"],
                                  row_name=df.loc[idx,"row_name"],
                                  rt=df.loc[idx,"RT"],
                                  sample_type=df.loc[idx,"type"],
                                  norm_int=df.loc[idx,"norm_intensity"]))
        
        return spot_list


    @staticmethod
    def backfill(spot_list:list,x,y,rad):
        """
        ## Description

        Backfills spots into a spotlist

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be backfilled|
        |x|float|x-coordinate of the spot to be backfilled|
        |y|float|y-coordinate of the spot to be backfilled|
        |rad|float|radius of the spot to be backfilled in pixels|

        ## Output

        None.
        """
        spot_list.append(spot(int(x),int(y),int(rad),note="Backfilled"))
    
    @staticmethod
    def find_topleft(spot_list:list):
        """
        ## Description

        Finds the top-left spot in a list of spot-objects.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be searched|
        
        ## Output

        spot-object
        """
        return sorted(spot_list,key=lambda s: s.x+s.y)[0]
    
    @staticmethod
    def find_topright(spot_list:list):
        """
        ## Description

        Finds the top-right spot in a list of spot-objects.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be searched|
        
        ## Output

        spot-object
        """
        return sorted(spot_list,key=lambda s: s.x-s.y)[-1]
    
    @staticmethod
    def sort_grid(spot_list:list,row_conv:dict=None,row_start:int=1,col_start:int=1, max_cycle:int=1000)->list:
        """
        ## Description

        Sorts a list of spots arranged in a grid and assigns row + column indices.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be sorted|
        |row_conv|dict|Dictionary containing info on row names|
        |row_start|int|Row-Index at which to start counting|
        |col_start|int|Column-Index at which to start counting|
        |max_cycle|int|Amount of cycles performed before breaking the loop, in case not all spots can be sorted|
        
        ## Output

        sorted list of spot-objects
        """
        
        # Create a copy of the spot-list to keep the original list unchanged.
        spot_copy=spot_list.copy()
        
        # Create new list containing sorted spots.
        sort_spots=[]
        
        # Repeat sequence for each row until all spots have been sorted.
        row_i=row_start
        while len(spot_copy)>0:
            # Find the top-left and top-right spots in the current spotlist.
            topleft=spot.find_topleft(spot_copy)
            tl_coord=np.array((topleft.x,topleft.y))
            
            topright=spot.find_topright(spot_copy)
            tr_coord=np.array((topright.x,topright.y))
            
            # Loop through all spots in the list and check if they are part of this row.
            current_row=[]
            col_i=col_start
            for s in spot_copy:
                # Calculate the distance of a spot to a line going through the top-left and top-right spot (top-row).
                dist_to_row=np.linalg.norm(np.cross(np.subtract(np.array((s.x,s.y)),tl_coord),np.subtract(tr_coord,tl_coord))/np.linalg.norm(np.subtract(tr_coord,tl_coord)))
                                
                # If the distance is smaller than the spots radius, the spot belongs to the current row.
                if dist_to_row<=s.rad:
                    # Append to current row.
                    current_row.append(s)
                    col_i+=1
            
            # Sort the row by the spots x-coordinates.
            current_row.sort(key=lambda s: s.x)
            # Add row and column indices to each spot of the current row.
            for s,col_idx in zip(current_row,range(col_start,col_i)):
                s.row=row_i
                s.col=col_idx

                if row_conv!=None:
                    s.row_name=row_conv[row_i].upper()
            
            # Add row to the sorted spot-list.
            sort_spots.extend(current_row)

            # Remove current row from spot_copy. list.remove() yielded bugs in the for loop so this approach is used.
            spot_copy=[s for s in spot_copy if s not in current_row]
            
            row_i+=1
            # Break if the maximum cycle number is exceeded.
            if row_i>max_cycle+row_start:
                print("Exceeded maximum cycle number!")
                break
        
        return sort_spots
    
    @staticmethod
    def normalize(spot_list:list) -> None:
        """
        ## Description

        Normalizes all spot-intensties using the labeled controls. 

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be normalized, must contain atleast one spot of the type "control"|
        """

        ctrl_mn=np.array([s.int for s in spot_list if s.type=="Control"]).mean()

        for s in spot_list:
            s.norm_int=s.int/ctrl_mn
    