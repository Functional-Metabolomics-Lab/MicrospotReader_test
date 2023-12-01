'''
-------------------------------------------------------------------------------
 Purpose:       Functions and Classes for detection of microspots.

 Author:        Simon Knoblauch
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
import scipy.signal as signal
import scipy.stats as stats

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

    if len(load.shape)<3:
        gray_img=load
    
    else:
        # Check if image is RGBA and convert image to grayscale.
        if load.shape[2]==4:
            gray_img=skimage.color.rgb2gray(load[:,:,0:3])

        # Convert RGB images to grayscale.
        elif load.shape[2]==3:
            gray_img=skimage.color.rgb2gray(load)
        
        else:
            gray_img=load

    # Invert the intensity values. Comment out if you do not wish to invert the image.
    if invert is True:
        gray_img=skimage.util.invert(gray_img)
    
    return gray_img

def baseline_correction(array,conv_lvl:float=0.001,conv_noise:float=0.0001,window_lvl:int=100,window_noise:int=5,poly_lvl:int=2,poly_noise:int=3):
    """
    ## Description
    Baseline correction of an input array using a modified version of the asymmetric least squares method.
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |array|Seq, Array|Sequence to be baseline corrected|
    |conv_lvl|float|convergence criteria for the determination of the baseline level|
    |conv_noise|float|convergence criteria for the determination of the baseline containing noise|
    |window_lvl|int|Window to be used for the savitzky-golay filter for detection of the baseline level|
    |window_noise|int|Window to be used for the savitzky-golay filter for detection of the baseline containing noise|
    |poly_lvl|int|order of the polynomial used to fit the data for baseline level detection|
    |poly_noise|int|order of the polynomial used to fit the data for detection of baseline containing noise|

    ## Returns
    Tuple of the values for the baseline aswell as the corrected baseline vales
    """

    # The algorithm is essentially performed twice: once to determine the level of the baseline and once to actually smooth the chromatogram. The first step is important as sometimes the savgol_filter would dip way below the actual baseline leading to artefact-peaks if the baseline-correction was just performed with that result.
    # This way the detected baseline cannot dip below the initally detected level, removing the artefact peaks.

    baseline_noise=array.copy()
    baseline_level=array.copy()

    # First time running the algo with a large window size to simply detect the general level of the baseline.
    rmsd_lvl=10
    while rmsd_lvl>conv_lvl:
        sg_filt=signal.savgol_filter(baseline_level,window_lvl,poly_lvl)
        baseline_new=np.minimum(sg_filt,baseline_level)
        rmsd_lvl=np.sqrt(np.mean((baseline_new-baseline_level)**2))
        baseline_level=baseline_new

    # Second time running the algo to actually be able to filter out the noise. 
    # Note that for the new baseline first the filter result is compared to the previous baseline to find the minimum, this way peaks are removed from the baseline. Then the result is compared to the coarse baseline level to find areas that deviate too much.
    rmsd_noise=10
    while rmsd_noise>conv_noise:
        sg_filt=signal.savgol_filter(baseline_noise,window_noise,poly_noise)
        # baseline_new=np.maximum(np.minimum(test,baseline_noise),baseline_level)
        baseline=np.minimum(sg_filt,baseline_noise)        
        baseline_new=[]
        for n,l in zip(baseline,baseline_level):
            if np.abs(n-l)>10*baseline_level.std():
                baseline_new.append(l)
            else:
                baseline_new.append(n)
        baseline_new=np.array(baseline_new)

        rmsd_noise=np.sqrt(np.mean((baseline_new-baseline_noise)**2))
        baseline_noise=baseline_new

    corr_ints=array-baseline_noise
    return baseline_noise,corr_ints

def baseline_noise(array:pd.Series, convergence_criteria:float=0.02):
    """
    ## Description
    Finds the standard deviation and mean of the baseline of a chromatogram
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |array|Seq, Array|Sequence for which the baseline noise should be determined|
    |convergence_criteria|float|convergence criteria for filtering outliers|
    
    ## Returns
    standard deviation of baseline
    mean value of baseline
    """
    mn_old=array.mean()
    std_old=array.std()

    rmsd=10
    while rmsd>convergence_criteria:
        test=array[array<mn_old+3*std_old]
        mn_new=test.mean()
        std_new=test.std()
        
        rmsd=np.sqrt(np.mean((std_new-std_old)**2))
        
        mn_old=mn_new
        std_old=std_new

    return std_old,mn_old

def peak_detection(df:pd.DataFrame,baseline_convergence:float=0.02,rel_height:float=0.95,min_dist:int=10,datacolumn_name:str="norm_intensity"):
    """
    ## Description
    Finds peaks and calculates the AUC in a spot-DataFrame
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |df|DataFrame|Spot-Dataframe to detect peaks in|
    |baseline_convergence|float|Cutoff rmsd value for filtering outliers for baseline determination|
    |rel_heigh|float|relative peak-height used as cutoff for AUC calculation|
    |datacolumn_name|str|name of the df column containing the y data for peak detection|
    
    ## Returns
    Dataframe containing information on peaks found in the spot-dataframe
    """
    bl_std,bl_mn=baseline_noise(df[datacolumn_name],baseline_convergence)
    
    peaks,_=signal.find_peaks(df[datacolumn_name],height=bl_mn+3*bl_std,distance=min_dist)

    _,_,left_ips,right_ips=signal.peak_widths(df[datacolumn_name],peaks,rel_height=rel_height)

    aft=pd.DataFrame(
            {
            "peak_idx":peaks,
            "RT":df.loc[peaks,"RT"].values,
            "start_idx":left_ips.astype("int32"),
            "end_idx":right_ips.astype("int32"),
            "RTstart":df.loc[left_ips.astype("int32"),"RT"].values,
            "RTend":df.loc[right_ips.astype("int32"),"RT"].values,
            "max_int":df.loc[peaks,datacolumn_name].values,
            "AUC":np.nan
            }
        ).rename_axis("peak_nr")

    for idx in aft.index:
        aft.loc[idx,"AUC"]=np.trapz(df.loc[aft.loc[idx,"start_idx"]:aft.loc[idx,"end_idx"],datacolumn_name])

    return aft

def annotate_mzml(exp:oms.MSExperiment(),spot_df:pd.DataFrame(),spot_mz:float, intensity_scalingfactor:float,norm_data:bool=True):
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
    |norm_data|bool|Uses Normalized Spot-Data if set to True|
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
            
            if norm_data==False:
                # Interpolate the spot intensity for the RT value
                interp_intensity=np.interp(rt_val,[prev_spot["RT"],next_spot["RT"]],[prev_spot["spot_intensity"],next_spot["spot_intensity"]])
            
            elif norm_data==True:
                # Interpolate the spot intensity for the RT value
                interp_intensity=np.interp(rt_val,[prev_spot["RT"],next_spot["RT"]],[prev_spot["norm_intensity"],next_spot["norm_intensity"]])
            
            if interp_intensity>0:
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

def feature_finding(exp:oms.MSExperiment,mass_error:float=10.0,noise_threshold:float=1000.0,min_fwhm=1.0,max_fwhm=60.0):
    exp.sortSpectra(True)

    mass_traces = []
    mtd = oms.MassTraceDetection()
    mtd_params = mtd.getDefaults()
    mtd_params.setValue(
        "mass_error_ppm", float(mass_error)
    )  # set according to your instrument mass error
    mtd_params.setValue(
        "noise_threshold_int", float(noise_threshold)
    )  # adjust to noise level in your data
    mtd.setParameters(mtd_params)
    mtd.run(exp, mass_traces, 0)

    mass_traces_split = []
    mass_traces_final = []
    epd = oms.ElutionPeakDetection()
    epd_params = epd.getDefaults()
    epd_params.setValue("width_filtering", "fixed")
    epd_params.setValue("min_fwhm",float(min_fwhm))
    epd_params.setValue("max_fwhm",float(max_fwhm))
    epd.setParameters(epd_params)
    epd.detectPeaks(mass_traces, mass_traces_split)

    if epd.getParameters().getValue("width_filtering") == "auto":
        epd.filterByPeakWidth(mass_traces_split, mass_traces_final)
    else:
        mass_traces_final = mass_traces_split

    fm = oms.FeatureMap()
    feat_chrom = []
    ffm = oms.FeatureFindingMetabo()
    ffm_params = ffm.getDefaults()
    ffm_params.setValue("isotope_filtering_model", "none")
    ffm_params.setValue(
        "remove_single_traces", "true"
    )  # set false to keep features with only one mass trace
    ffm_params.setValue("mz_scoring_by_elements", "false")
    ffm_params.setValue("report_convex_hulls", "true")
    ffm.setParameters(ffm_params)
    ffm.run(mass_traces_final, fm, feat_chrom)

    fm.setUniqueIds()
    ft=fm.get_df()

    return ft[["charge","RT","mz","RTstart","RTend","MZstart","MZend","quality","intensity"]]

def xic_generator(exp:oms.MSExperiment, ft:pd.DataFrame):
    """
    ## Description

    Creates feature-XICs and stores them in a dictionary as DataFrames from an oms.MSExperiment object. 
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |ft|DataFrame|Feature table containing information on mz-value and retention time. retention time column must be called "RT"|
    |exp|oms.MSExperiment|MSExperiment object containing the LC-MS run from which the features were determined|

    ## Output


    """
    specs={spec.getRT():{"mz":spec.get_peaks()[0],"int":spec.get_peaks()[1]} for spec in exp if spec.getMSLevel()==1}

    xics={}
    for i in ft.index:
        
        intsum_list=[]
        rtlist=[]

        for rt,pk in specs.items():

            if rt >= ft.loc[i,"RTstart"] and rt <= ft.loc[i,"RTend"]:

                intsum_list.append(pk["int"][(ft.loc[i,"MZstart"]<=pk["mz"]) & (ft.loc[i,"MZend"]>=pk["mz"])].sum())
                rtlist.append(rt)

        xics[i]=pd.DataFrame({"rt":rtlist,"int":intsum_list})
    
    return xics

def extract_xic_peakwindow(xic_dict:dict,ft:pd.DataFrame,window:float):
    """
    ## Description

    Calculates the pearson correlation coefficient of a normalized activity peak to a normalized feature-peak correlated to the activity peak by retention time.
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |ft|DataFrame|Feature table containing information on mz-value and retention time. retention time column must be called "RT"|
    |xic_dict|dict|Dictionary containing the xics of all features|
    |window|float|window in [s] around the peak that is extracted|
    
    ## Output

    Dictionary containing DataFrames containing the chromatograms of the peaks extracted from the feature-chromatograms.
    """
    cutxic={}
    for i, df in xic_dict.items():
        df_cut=df.loc[(df["rt"]>ft.loc[i,"RT"]-0.5*window)&(df["rt"]<ft.loc[i,"RT"]+0.5*window)].copy()
        df_cut["int"]=df_cut.loc[:,"int"]/(df_cut.loc[:,"int"].max())
        cutxic[i]=df_cut
    return cutxic

def peak_pearsoncorr(xic_dict,ft,ap_df,peak,idx,ydata_name):
    """
    ## Description

    Calculates the pearson correlation coefficient of a normalized activity peak to a normalized feature-peak correlated to the activity peak by retention time.
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |ft|DataFrame|Feature table containing information on mz-value and retention time. retention time column must be called "RT"|
    |ap_df|DataFrame|Activity peak table, contains information on activity peaks at specific retention times|
    |peak|DataFrame|Activity peak used during the correlation|
    |xic_dict|dict|Dictionary containing the xics of all features|
    |idx|int|index of the current row in the feature table|
    |ydata_name|str|Name of the Column in act_df containing the y-axis information of the activity chromatogram|
    """

    for i in ft.loc[ft[f"corr_activity_peak{idx}"]>0].index:
        df=xic_dict[i]
        if len(df)>=len(ap_df):
            interpRT=np.linspace(peak.RTstart,peak.RTend,len(df))
            interpInt=np.interp(interpRT,ap_df["RT"],ap_df[ydata_name])
            ft.loc[i,f"pearson_corr_peak{idx}"]=stats.pearsonr(df["int"],interpInt).statistic
        else:
            interpRT=np.linspace(df.rt.iloc[0],df.rt.iloc[-1],len(ap_df))
            interpInt=np.interp(interpRT,df.rt,df["int"])
            ft.loc[i,f"pearson_corr_peak{idx}"]=stats.pearsonr(interpInt,ap_df[ydata_name]).statistic

def peakshape_corr(xic_dict:dict,ft:pd.DataFrame,ap_df:pd.DataFrame,act_df:pd.DataFrame,idx:int,ydata_name:str="norm_intensity"):
    """
    ## Description

    Calculates the pearson correlation coefficient of each normalized activity peak to all normalized feature-peaks correlated to the activity peak by retention time.
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |ft|DataFrame|Feature table containing information on mz-value and retention time. retention time column must be called "RT"|
    |ap_df|DataFrame|Activity peak table, contains information on activity peaks at specific retention times|
    |act_df|DataFrame|Dataframe resulting from the data merging step of the microspot reader workflow|
    |xic_dict|dict|Dictionary containing the xics of all features|
    |idx|int|index of the current row in the feature table|
    |ydata_name|str|Name of the Column in act_df containing the y-axis information of the activity chromatogram|
    """
    pk=ap_df.loc[idx].copy()

    cutap=act_df.loc[pk["start_idx"]:pk["end_idx"]].copy()
    cutap[ydata_name]=cutap[ydata_name]/cutap[ydata_name].max()
    width=np.abs(pk["RTend"]-pk["RTstart"])

    cutxic=extract_xic_peakwindow(xic_dict,ft,width)
    
    peak_pearsoncorr(cutxic,ft,cutap,pk,idx,ydata_name)

def activity_annotation_features(ft:pd.DataFrame,aft:pd.DataFrame,act_df:pd.DataFrame,xic_dict:dict,rt_tolerance:float,ydata_name:str="norm_intensity"):
    """
    ## Description

    Annotates the feature table from metabolomics feature detection with activity data from an activity feature table
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |ft|DataFrame|Feature table containing information on mz-value and retention time. retention time column must be called "RT"|
    |aft|DataFrame|Activity peak table, contains information on activity peaks at specific retention times|
    |act_df|DataFrame|Dataframe resulting from the data merging step of the microspot reader workflow|
    |xic_dict|dict|Dictionary containing the xics of all features|
    |rt_tolerance|float|tolerance of rt in seconds within which features will be correlated|
    |ydata_name|str|Name of the Column in act_df containing the y-axis information of the activity chromatogram|
    """
    for iat in aft.index:
        ft.loc[rt_tolerance>=np.abs(ft["RT"]-aft.loc[iat,"RT"]),f"corr_activity_peak{iat}"]=aft.loc[iat,"AUC"]

        peakshape_corr(xic_dict,ft,aft,act_df,iat,ydata_name)

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
    def detect(img,min_rad:int=40,max_rad:int=100,min_xdist:int=70,min_ydist:int=70,thresh:float=0.2,min_obj_size:int=800):
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
        skel=skimage.morphology.binary_dilation(skel,skimage.morphology.disk(3))

        test_radii=np.arange(min_rad,max_rad+1)
        # Circle detection by hough transform.
        halo_hough=skimage.transform.hough_circle(skel,test_radii)
        accums, cx, cy, radii=skimage.transform.hough_circle_peaks(halo_hough,test_radii, 
                                                                    min_xdistance=min_xdist,
                                                                    min_ydistance=min_ydist,
                                                                    threshold=thresh*halo_hough.max())

        halo_list=[halo(x,y,rad) for x,y,rad in zip(cx,cy,radii)]
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
 