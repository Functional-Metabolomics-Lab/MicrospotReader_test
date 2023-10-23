'''
-------------------------------------------------------------------------------
 Purpose:       Functions and Classes for detection of microspots.

 Author:        Simon Knoblauch

 Last modified: 2023-09-27
-------------------------------------------------------------------------------
'''

# Importing dependencies
import imageio.v3 as iio
import pandas as pd
import skimage
import numpy as np
from pathlib import Path
import streamlit as st
import pyopenms as oms

@st.cache_data
def conv_gridinfo(point1:str,point2:str,conv_dict:dict) -> dict:
    """
    ## Description:
    Takes in information on the first and last point on a uniformly spaced grid and converts it into the grids properties.

    ## Input:

    |Parameter|Type|Description|
    |---|---|---|
    |point1|str|First Point on the grid. The first character determines row, the rest determine column.|
    |point2|str|Last Point on the grid. The first character determines row, the rest determine column.|
    |conv_dict|dict|Dictionary by which the row characters are converted to numeric values.|

    ## Output:

    Dictionary of grid properties.
    """

    # Determining the indexes of the first and last row
    firstrow_nr=conv_dict[point1[0].lower()]
    lastrow_nr=conv_dict[point2[0].lower()]
    rowcount=lastrow_nr-firstrow_nr+1

    # Determining the indexes of the first and last column
    firstcol_nr=int(point1[1:])
    lastcol_nr=int(point2[1:])
    colcount=lastcol_nr-firstcol_nr+1

    # Calculating the number of total spots
    nr_spots=rowcount*colcount

    # Storing results in Dictionary.
    grid_properties={
        "rows":{
            "bounds":[firstrow_nr,lastrow_nr],
            "length":rowcount},
        "columns":{
            "bounds":[firstcol_nr,lastcol_nr],
            "length":colcount},
        "spot_nr":nr_spots}
    
    return grid_properties

@st.cache_data
def prep_img(filename:Path,invert:bool=False) -> np.array:
    """
    ## Description
    Loads a grayscale, RGB or RGBA image from a filepath and returns a grayscale array of the image.
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |filename|Path, Str|Relative or absolute path of the image to be loaded.|
    |invert|bool|Set True if the image should be inverted.|

    ## Output

    Grayscale Numpy Array of the image.

    """
    
    # Read image file.
    load=iio.imread(filename)

    # Check if image is RGBA and convert image to grayscale.
    if load.shape[2]==4:
        gray_img=skimage.color.rgb2gray(load[:,:,0:3])

    # Convert RGB images to grayscale.
    elif load.shape[2]==3:
        gray_img=skimage.color.rgb2gray(load)
    
    else:
        gray_img=load

    # Invert the intensity values. Comment out if you do not wish to invert the image.
    if invert:
        gray_img=skimage.util.invert(gray_img)
    
    return gray_img


def annotate_mzml(exp,spot_df,spot_mz, intensity_scalingfactor):
    """
    ## Description
    Sets the value of a specified m/z in an MS1 spectrum at a specific retention time to a scaled and interpolated spot-intensity value based off of a DataFrame containing RT-matched spot intensities.
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |exp|MSExperiment|pyOpenMS MSExperiment class with a loaded mzml file|
    |spot_df|DataFrame|DataFrame containing Retention Time matched spot-intensities|
    |spot_mz|float|m/z value to be set to the interpolated spot-intensity|
    |intensity_scalingfactor|float|Value by which to scale the interpolated spotintensity for MS1 annotation|
    """
    spots=spot_df.sort_values("RT")

    spec_list=[]
    rt_list=[]
    int_list=[]
    # Loop through all Spectra in the mzml file.
    for spectrum in exp:
        # Check if the spectrum is MS1
        if spectrum.getMSLevel()==1:

            # Get the RT
            rt_val=spectrum.getRT()
            
            # Index the spot with the closest, shorter RT value compared to the RT of the spectrum.
            try:
                prev_spot=spots[spots["RT"]<=rt_val].iloc[-1]
            except:
                prev_spot=spots.iloc[0]
            
            # Index the spot with the closest, longer RT value compared to the RT of the spectrum
            try:        
                next_spot=spots[spots["RT"]>rt_val].iloc[0]
            except:
                # If there is no higher RT in the spotlist, take the next smallest one.
                next_spot=prev_spot
            
            # Interpolate the spot intensity for the RT value
            interp_intensity=np.interp(rt_val,[prev_spot["RT"],next_spot["RT"]],[prev_spot["spot_intensity"],next_spot["spot_intensity"]])
            
            # Append the array of peak-m/z values with the one specified to save the spot intensity
            peak_mz=np.append(spectrum.get_peaks()[0],spot_mz)
            # Append the array of peak-intensities with the scaled version of the interpolated spot intensity
            peak_int=np.append(spectrum.get_peaks()[1],interp_intensity*intensity_scalingfactor)
            # Save the new peak arrays in the spectrum.
            spectrum.set_peaks((peak_mz,peak_int))
        
        # Append current spectrum to the modified list of spectra
        spec_list.append(spectrum)
        # Append values for use in chromatogramm plot
        rt_list.append(rt_val)
        int_list.append(interp_intensity)

    # Save the spectra-list to the MS Experiment
    exp.setSpectra(spec_list)


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
    def __init__(self,x:float,y:float,rad:int=25,halo_rad=np.nan,int=np.nan,note="Initial Detection",row:int=np.nan,col:int=np.nan,row_name:str=np.nan,rt=np.nan,sample_type:str="Sample",norm_int:float=np.nan) -> None:
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
    def detect(gray_img:np.array,spot_nr:int,canny_sig:int=10,canny_lowthresh:float=0.001,canny_highthresh:float=0.001,hough_minx:int=70,hough_miny:int=70,hough_thresh:float=0.3,small_rad:int=20,large_rad:int=30) -> list:
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
        
        return spotlist
    
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
    def backfill(spot_list:list,x,y):
        """
        ## Description

        Backfills spots into a spotlist

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |spot_list|list|List of spot-objects to be backfilled|
        |x|float|x-coordinate of the spot to be backfilled|
        |y|float|y-coordinate of the spot to be backfilled

        ## Output

        None.
        """
        spot_list.append(spot(int(x),int(y),note="Backfilled"))
    
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
    
class gridpoint:
    """
    ## Description

    Class containing information on points lying on a grid.

    ## __init__ Input:

    |Parameter|Type|Description|
    |---|---|---|
    |x|float|X-coordinate of the gridpoint|
    |y|float|Y-coordinate of the gridpoint|

    ## Class Attributes

    |Name|Description|
    |---|---|
    |x|X-coordinate of the gridpoint|
    |y|Y-coordinate of the gridpoint|
    |min_dist|Minimum distance of the gridpoint to a list of points. Default value is 100000.|
    """
    def __init__(self,x,y) -> None:
        self.x=x
        self.y=y
        self.min_dist=10000

    def eval_distance(self,spot_x,spot_y):
        """
        ## Description

        Calculates the distance between the gridpoint and another given point. 
        If the distance is smaller than the current min_dist, assign the new distance to min_dist.

        ## Input
        |Parameter|Type|Description|
        |---|---|---|
        |spot_x|float|X-coordinate of the other point|
        |spot_y|float|Y-coordinate of the other point|

        ## Output

        Distance between the 2 points
        """

        # Calculate the euclidean distance between the 2 points
        pointdist=np.linalg.norm(np.array((self.x,self.y))-np.array((spot_x,spot_y)))
        
        # Add the distance to the current gridpoint if it is smaller than all previously tested gridpoints.
        if pointdist<self.min_dist: 
            self.min_dist=pointdist
        
        return pointdist

class gridline:
    """
    ## Description

    Class containing information on lines lying on a grid.

    ## __init__ Input:

    |Parameter|Type|Description|
    |---|---|---|
    |angle|float|angle of the line in radians.|
    |distance|float|distance of the line from the origin|

    ## Class Attributes

    |Name|Description|
    |---|---|
    |dist|distance of the line from the origin|
    |angle|angle of the line in radians|
    |slope|slope of the line|
    |y_int|y-intercept of the line|
    |alignment|"hor" for horizontally and "vert" for vertically aligned lines|
    """

    def __init__(self, angle, distance):
        self.dist=distance
        self.angle=angle
        self.slope=np.tan(angle+np.pi/2)
        x0,y0=distance*np.array([np.cos(angle),np.sin(angle)])
        self.y_int=y0-self.slope*x0
        self.alignment=np.nan
    
    def __repr__(self):
        return f"y={self.slope:.2f}*x+{self.y_int:.2f}"

    def intersect(self,line2) -> tuple:
        """
        ## Description

        Calculates the intersect between the current line and another line object.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |line2|gridline|line to calculate the intersect with|

        ## Output

        Tuple containing the x and y coordinate of the line intersection.
        """
        
        # Calculation of the x and y coordinates of the line intersection.
        x=(line2.y_int-self.y_int)/(self.slope-line2.slope)
        y=self.slope*x+self.y_int
        
        point=gridpoint(x,y)
        return point
    
    @staticmethod
    @st.cache_data
    def detect(img:np.array,max_tilt:int=5,min_dist:int=80,threshold:float=0.2) -> list:
        """
        ## Description

        Determines lines lying on a grid from an image containing gridpoints.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |img|Array|np.array of an image containing gridpoints|
        |max_tilt|int|Maximum allowed tilt of the grid in degrees.|
        |min_dist|int|Minumum distance between two detected lines|
        |threshold|float|Fraction of max|

        ## Output

        List of gridline objects.
        """

        # Create a hough-transform of the original image
        line_img,ang,dist=skimage.transform.hough_line(img)
        
        # Set the intensites of all lines with unwanted angles to 0.
        line_img[:,np.r_[max_tilt:89-max_tilt,91+max_tilt:180-max_tilt]]=0
        
        # Detect lines in the hough transformed image.
        accum,angle,distance=skimage.transform.hough_line_peaks(line_img,ang,dist,min_distance=min_dist,threshold=threshold*line_img.max())
        
        # Saving all gridlines in a list
        gridlines=[gridline(a,d) for a,d in zip(angle,distance)]
        
        # Assigning horizontal and vertical lines.
        for line in gridlines: 
            if np.abs(np.rad2deg(line.angle))<=max_tilt: 
                line.alignment="hor"
            elif np.abs(np.rad2deg(line.angle))>=90-max_tilt: 
                line.alignment="vert"   
                
        return gridlines

class halo:
    """
    ## Description

    Class containing information on halos detected during halo-detection

    ## __init__ Input:

    |Parameter|Type|Description|
    |---|---|---|
    |x|float|x-coordinate of the halo|
    |y|float|y-coordinate of the halo|
    |rad|int|radius of the spot|

    ## Class Attributes

    |Name|Description|
    |---|---|
    |x|x-coordinate of the halo|
    |y|y-coordinate of the halo|
    |rad|radius of the halo|
    """
    def __init__(self,x,y,rad) -> None:
        self.x=x
        self.y=y
        self.rad=rad

    @staticmethod
    @st.cache_data
    def detect(img,canny_sig:float=3.52941866,canny_lowthresh:float=44.78445877,canny_highthresh:float=44.78445877,hough_minx:int=70,hough_miny:int=70,hough_thresh:float=0.38546213,min_rad:int=40,max_rad:int=70):
        """
        ## Description

        Crude detection of halos in a grayscale image.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |img|np.array|Grayscale np.array of image to be analyzed|
        |canny_sig|int|Sigma value for gaussian blur during canny edge detection|
        |canny_lowthresh|float|Low threshold for canny edge detection|
        |canny_highthres|float|High threshold for canny edge detection|
        |hough_minx|int|Miniumum distance in x direction between 2 peaks during circle detection using hough transform|
        |hough_miny|int|Miniumum distance in y direction between 2 peaks during circle detection using hough transform|
        |hough_thresh|int|threshold of peak-intensity during circle detection using hough transform as fraction of maximum value|
        |min_rad|int|Minimum tested radius|
        |max_rad|int|Maximum tested radius|

        ## Output

        List of detected halos as halo-objects
        """
        
        # Create a mask of the image only containing halos.
        halo_mask=img<skimage.filters.threshold_yen(img)

        # Applying the mask to the histeq_img yields more consistent results for circle detection. The mask itself yields better results when calculated from the raw image.
        halo_img=skimage.filters.rank.equalize(skimage.util.img_as_ubyte(img),skimage.morphology.disk(50))
        halo_img[halo_mask]=0

        # Canny edge detection and follow up circle detection using hough transform.
        halo_edge=skimage.feature.canny(halo_img,canny_sig,canny_lowthresh,canny_highthresh)
        halo_radii=np.arange(min_rad,max_rad+1) # Radii tested for.
        halo_hough=skimage.transform.hough_circle(halo_edge,halo_radii)

        h_accum,h_x,h_y,h_radii=skimage.transform.hough_circle_peaks(
            halo_hough,
            halo_radii,
            min_xdistance=hough_minx,
            min_ydistance=hough_miny,
            threshold=hough_thresh*halo_hough.max()
            )
        
        halo_list=[halo(x,y,rad) for x,y,rad in zip(h_x,h_y,h_radii)]
        return halo_list
    
    @staticmethod
    def create_df(halo_list:list) -> pd.DataFrame:
        """
        ## Description

        Creates a DataFrame from a list of halos.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |halo_list|list|List of halo-objects to be turned into a DataFrame|

        ## Output

        DataFrame of halo-list.
        """

        halo_df=pd.DataFrame({"x_coord":[i_spot.x for i_spot in halo_list],
                              "y_coord":[i_spot.y for i_spot in halo_list],
                              "radius":[i_spot.rad for i_spot in halo_list]})
        
        return halo_df
 