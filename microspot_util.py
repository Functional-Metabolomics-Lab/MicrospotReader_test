# Importing dependencies
import imageio.v3 as iio
import pandas as pd
import skimage
import numpy as np
from pathlib import Path

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
    """
    def __init__(self,x:float,y:float,rad:int=25,halo_rad=np.nan,int=np.nan,note="Initial Detection",row:int=np.nan,col:int=np.nan,row_name:str=np.nan) -> None:
        self.x=x
        self.y=y
        self.rad=rad
        self.halo=halo_rad
        self.int=int
        self.note=note
        self.row=row
        self.col=col
        self.row_name=row_name
    
    def assign_halo(self,halo_list:list) -> None:
        """
        ## Description
        Checks a list of halos for a match and assigns it to the spot.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |halo_list|int|List of halo-objects|
        """
        for h in halo_list:
            if np.linalg.norm(np.array((h.x,h.y))-np.array((self.x,self.y)))<14.73402725:
                self.halo=h.rad
            

    def get_intensity(self,img:np.array) -> None:
        """
        ## Description
        Determines the average pixel-intensity of a spot in an image.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |img|np.array|Image to extract spot-intensity from|
        
        ## Output

        Avgerage intensity of pixels in spot.
        """
        try:
            # Indices of all pixels part of the current spot
            rr,cc=skimage.draw.disk((self.y,self.x),self.rad)
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
                                             "x_coord":self.x,
                                             "y_coord":self.y,
                                             "radius":self.rad,
                                             "halo":self.halo,
                                             "spot_intensity:":self.int,
                                             "note":self.note}).to_frame().T],ignore_index=True)
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
    def detect(gray_img:np.array,spot_nr:int) -> list:
        """
        ## Description

        Crude detection of spots in a grayscale image.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |gray_img|np.array|Grayscale np.array of image to be analyzed|
        |spot_nr|int|Maximum number of spots to be detected in the image|

        ## Output

        List of detected spots as spot-objects
        """

        histeq_img=skimage.filters.rank.equalize(skimage.util.img_as_ubyte(gray_img),skimage.morphology.disk(50))
        edges=skimage.feature.canny(
        image=histeq_img,
        sigma=10,
        low_threshold=0.001,
        high_threshold=0.001
        )

        # Range of Radii that are tested during inital spotdetection.
        tested_radii=np.arange(20,31)

        # Hough transform for a circle of the edge-image and peak detection to find circles in earlier defined range of radii.
        spot_hough=skimage.transform.hough_circle(edges,tested_radii)
        accums,spot_x,spot_y,spot_rad=skimage.transform.hough_circle_peaks(
            hspaces=spot_hough,
            radii=tested_radii,
            total_num_peaks=spot_nr,
            min_xdistance=70,
            min_ydistance=70,
            threshold=0.3*spot_hough.max()
            )
        
        spotlist=[spot(x,y,rad) for x,y,rad in zip(spot_x,spot_y,spot_rad)]
        
        return spotlist
    
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
                              "x_coord":[i_spot.x for i_spot in spot_list],
                              "y_coord":[i_spot.y for i_spot in spot_list],
                              "radius":[i_spot.rad for i_spot in spot_list],
                              "halo":[i_spot.halo for i_spot in spot_list],
                              "spot_intensity":[i_spot.int for i_spot in spot_list],
                              "note":[i_spot.note for i_spot in spot_list]})
        return spot_df
    
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
    def detect(img:np.array,max_tilt:int=5) -> list:
        """
        ## Description

        Determines lines lying on a grid from an image containing gridpoints.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |img|Array|np.array of an image containing gridpoints|
        |max_tilt|int|Maximum allowed tilt of the grid in degrees.|

        ## Output

        List of gridline objects.
        """

        # Create a hough-transform of the original image
        line_img,ang,dist=skimage.transform.hough_line(img)
        
        # Set the intensites of all lines with unwanted angles to 0.
        line_img[:,np.r_[max_tilt:89-max_tilt,91+max_tilt:180-max_tilt]]=0
        
        # Detect lines in the hough transformed image.
        accum,angle,distance=skimage.transform.hough_line_peaks(line_img,ang,dist,min_distance=80,threshold=0.2*line_img.max())
        
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
    def detect(img):
        """
        ## Description

        Crude detection of halos in a grayscale image.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |img|np.array|Grayscale np.array of image to be analyzed|

        ## Output

        List of detected halos as halo-objects
        """
        
        # Create a mask of the image only containing halos.
        halo_mask=img<skimage.filters.threshold_yen(img)

        # Applying the mask to the histeq_img yields more consistent results for circle detection. The mask itself yields better results when calculated from the raw image.
        halo_img=skimage.filters.rank.equalize(skimage.util.img_as_ubyte(img),skimage.morphology.disk(50))
        halo_img[halo_mask]=0

        # Canny edge detection and follow up circle detection using hough transform.
        halo_edge=skimage.feature.canny(halo_img,3.52941866,44.78445877,44.78445877)
        halo_radii=np.arange(40,70) # Radii tested for.
        halo_hough=skimage.transform.hough_circle(halo_edge,halo_radii)

        h_accum,h_x,h_y,h_radii=skimage.transform.hough_circle_peaks(
            halo_hough,
            halo_radii,
            min_xdistance=70,
            min_ydistance=70,
            threshold=0.38546213*halo_hough.max()
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
 