# Importing dependencies
import pandas as pd
import skimage
import numpy as np
import streamlit as st

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
    def detect_old(img,canny_sig:float=3.52941866,canny_lowthresh:float=44.78445877,canny_highthresh:float=44.78445877,hough_minx:int=70,hough_miny:int=70,hough_thresh:float=0.38546213,min_rad:int=40,max_rad:int=70):
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
    @st.cache_data
    def detect(img,min_rad:int=40,max_rad:int=100,min_xdist:int=70,min_ydist:int=70,thresh:float=0.2,min_obj_size:int=800,troubleshoot:bool=False,dil_disk:int=3):
        """
        ## Description

        Crude detection of halos in a grayscale image.

        ## Input

        |Parameter|Type|Description|
        |---|---|---|
        |img|np.array|Grayscale np.array of image to be analyzed|
        |min_obj_size|int|Minimum size of objects after creation of mask, anything smaller will be removed|
        |min_xdist|int|Miniumum distance in x direction between 2 peaks during circle detection using hough transform|
        |min_ydist|int|Miniumum distance in y direction between 2 peaks during circle detection using hough transform|
        |thresh|int|threshold of peak-intensity during circle detection using hough transform as fraction of maximum value|
        |min_rad|int|Minimum tested radius|
        |max_rad|int|Maximum tested radius|

        ## Output

        List of detected halos as halo-objects
        """

        # Perform morphological reconstruction
        mask=np.copy(img)
        seed=np.copy(img)
        seed[1:-1,1:-1]=img.min()
        dilated=skimage.morphology.reconstruction(seed,mask,method="dilation")
        recon=img-dilated
        
        # Threshold the reconstructed image and remove noise
        bin_recon=recon>skimage.filters.threshold_otsu(recon)
        bin_recon=skimage.morphology.remove_small_objects(bin_recon,min_size=min_obj_size)
        # Biary opening to open holes for halos containing spots
        bin_recon=skimage.morphology.binary_opening(bin_recon,skimage.morphology.disk(5))

        # Generate a Skeleton of the binary reconstruction to get a single circle for each halo
        skel=skimage.morphology.skeletonize(bin_recon)
        # Dilate the skeleton to have some tolerance for circle detection
        skel=skimage.morphology.binary_dilation(skel,skimage.morphology.disk(dil_disk))

        test_radii=np.arange(min_rad,max_rad+1)
        # Circle detection by hough transform.
        halo_hough=skimage.transform.hough_circle(skel,test_radii)
        accums, cx, cy, radii=skimage.transform.hough_circle_peaks(halo_hough,test_radii, 
                                                                    min_xdistance=min_xdist,
                                                                    min_ydistance=min_ydist,
                                                                    threshold=thresh*halo_hough.max())

        halo_list=[halo(x,y,rad) for x,y,rad in zip(cx,cy,radii)]
        
        if troubleshoot is True:
            return halo_list, recon, bin_recon, skel, halo_hough
        else:
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
 