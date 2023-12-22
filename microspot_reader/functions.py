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

def baseline_correction2(array,conv_lvl=0.001,window_lvl=100,poly_lvl=2):
    """
    ## Description
    Baseline correction of an input array using the savitz golay filter.
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |array|Seq, Array|Sequence to be baseline corrected|
    |conv_lvl|float|convergence criteria for the determination of the baseline level||
    |window_lvl|int|Window to be used for the savitzky-golay filter for detection of the baseline level|
    |poly_lvl|int|order of the polynomial used to fit the data for baseline level detection|

    ## Returns
    Tuple of the values for the baseline aswell as the corrected baseline vales
    """
    baseline_level=array.copy()
    
    if len(baseline_level)<window_lvl:
        window_lvl=len(baseline_level)

    # First time running the algo with a large window size to simply detect the general level of the baseline.
    rmsd_lvl=10
    while rmsd_lvl>conv_lvl:
        sg_filt=signal.savgol_filter(baseline_level,window_lvl,poly_lvl)
        baseline_new=np.minimum(sg_filt,baseline_level)
        rmsd_lvl=np.sqrt(np.mean((baseline_new-baseline_level)**2))
        baseline_level=baseline_new

    return baseline_level, array-baseline_level
    
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
    
    if len(baseline_level)<window_lvl:
        window_lvl=len(baseline_level)

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
    return baseline_noise,baseline_level,corr_ints

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
    # Loop through the algorithm until the rmsd of the standard deviation is below a defined criterium
    while rmsd>convergence_criteria:#
        # exclude values in the array that are most likely outliers -> ie peaks
        test=array[array<mn_old+3*std_old]
        # calculate the new mean and std
        mn_new=test.mean()
        std_new=test.std()
        # caluclate rmsd
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
    # *** Not used as this does 1d peak detection ***
    bl_std,bl_mn=baseline_noise(df[datacolumn_name],baseline_convergence)

    min_height=bl_mn+3*bl_std

    peaks,_=signal.find_peaks(df[datacolumn_name],height=min_height,distance=min_dist,width=2)

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

def img_peak_detection(df:pd.DataFrame,datacolumn_name:str="smoothed_int",threshold:float=0.0):
    """
    ## Description
    Finds peaks and calculates the AUC in a spot-DataFrame
    
    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |df|DataFrame|Spot-Dataframe to detect peaks in|
    |datacolumn_name|str|name of the df column containing the y data for peak detection|
    
    ## Returns
    Dataframe containing information on peaks found in the spot-dataframe and updated main dataframe
    """
    # Create a heatmap from the spot intensities, it is better to do peak detection in 2d instead of on the chromatogram due to artifacts during image capturing.
    img=df.pivot_table(datacolumn_name,index="row_name",columns="column")

    # do 2d peak detection 
    peaks=skimage.feature.peak_local_max(
        image=img.to_numpy(),
        min_distance=1,
        exclude_border=False,
        threshold_abs=threshold
    )

    # get the main df indexes of all minimas to figure out peak width
    minima,_=signal.find_peaks(-df[datacolumn_name].to_numpy())
    minima=df.index[minima]

    # get the main df indexes of all peaks
    peak_idx=[df.loc[(df["row_name"]==img.index[p[0]])&(df["column"]==img.columns[p[1]])].index.item() for p in peaks]

    # get the left most indexes of all detected peaks
    left_ips=[minima[minima<i][-1] if any(minima<i) else df.index[0] for i in peak_idx]

    # get the right most indexes of all detected peaks
    right_ips=[minima[minima>i][0] if any(minima>i) else df.index[-1] for i in peak_idx]

    # save peak data in a new dataframe
    aft=pd.DataFrame(
                {
                "peak_idx":peak_idx,
                "RT":df.loc[peak_idx,"RT"].values,
                "start_idx":left_ips,
                "end_idx":right_ips,
                "RTstart":df.loc[left_ips,"RT"].values,
                "RTend":df.loc[right_ips,"RT"].values,
                "max_int":df.loc[peak_idx,datacolumn_name].values,
                "AUC":np.nan
                }
            ).rename_axis("peak_nr")

    # calculate the AUC of each peak
    for idx in aft.index:
        aft.loc[idx,"AUC"]=np.trapz(df.loc[aft.loc[idx,"start_idx"]:aft.loc[idx,"end_idx"],datacolumn_name])

    return df,aft,peaks

def annotate_mzml(exp:oms.MSExperiment,spot_df:pd.DataFrame,spot_mz:float, intensity_scalingfactor:float,norm_data:bool=True):
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

def feature_finding(exp:oms.MSExperiment,filename:str,mass_error:float=10.0,noise_threshold:float=1000.0,min_fwhm=1.0,max_fwhm=60.0):
    """
    ## Description
    Implemented feature finding algorithm from pyopenms https://pyopenms.readthedocs.io/en/latest/user_guide/feature_detection.html

    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |exp|MSExperiment|pyOpenMS MSExperiment class with a loaded mzml file|
    |mass_error|float|mass error in ppm|
    |noise_threshold|float|intensity threshold for noise|
    |min_fwhm|float|Minimum full width at half maximum of features|
    |max_fwhm|float|Maximum full width at half maximum of features|

    ## Output

    oms.FeatureMap instance containing information on the detected features
    """
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
    fm.setPrimaryMSRunPath([filename.encode()])

    return fm

def ms2_mapping(exp:oms.MSExperiment,fm:oms.FeatureMap):
    """
    ## Description
    Implemented algorithm for mapping ms2 data to features from pyopenms https://pyopenms.readthedocs.io/en/latest/user_guide/untargeted_metabolomics_preprocessing.html

    Maps the ms2 data to features in the featuremap

    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |exp|MSExperiment|pyOpenMS MSExperiment class with a loaded mzml file|
    |fm|oms.FeatureMap|Featuremap instance from feature finding|
    """

    use_centroid_rt = False
    use_centroid_mz = True
    mapper = oms.IDMapper()
    peptide_ids = []
    protein_ids = []

    mapper.annotate(
        fm,
        peptide_ids,
        protein_ids,
        use_centroid_rt,
        use_centroid_mz,
        exp,
    )

def adduct_detector(fm:oms.FeatureMap,adduct_list:list[str]=[b'H:+:0.4', b'Na:+:0.2', b'NH4:+:0.2', b'H3O1:+:0.1', b'CH2O2:+:0.1',b"H-2O-1:0:0.2"]):
    """
    ## Description
    Implemented algorithm adduct detection from pyopenms https://pyopenms.readthedocs.io/en/latest/user_guide/adduct_detection.html

    ## Input

    |Parameter|Type|Description|
    |---|---|---|
    |fm|oms.FeatureMap|Featuremap instance from feature finding|
    |adduct_list|list of strings|list of strings containing all expected adducts, rules for the list can be found on the linked website|

    ## Output
    ft -> pd.Dataframe feature table containing information on all features
    groups -> result consensus map: will store grouped features belonging to a charge group, used to save an mgf file
    """
    
    # initialize MetaboliteFeatureDeconvolution
    mfd = oms.MetaboliteFeatureDeconvolution()

    # get default parameters
    params = mfd.getDefaults()
    # update/explain most important parameters

    # adducts to expect: elements, charge and probability separated by colon
    # the total probability of all charged adducts needs to be 1
    # e.g. positive mode:
    # proton dduct "H:+:0.6", sodium adduct "Na:+:0.4" and neutral water loss "H-2O-1:0:0.2"
    # e.g. negative mode:
    # with neutral formic acid adduct: "H-1:-:1", "CH2O2:0:0.5"
    # multiples don't need to be specified separately:
    # e.g. [M+H2]2+ and double water loss will be detected as well!
    # optionally, retention time shifts caused by adducts can be added
    # e.g. a formic acid adduct causes 3 seconds earlier elution "CH2O2:0:0.5:-3"
    params.setValue(
        "potential_adducts", 
        adduct_list
    )

    # expected charge range
    # e.g. for positive mode metabolomics:
    # minimum of 1, maximum of 3, maximum charge span for a single feature 3
    # for negative mode:
    # charge_min = -3, charge_max = -1
    params.setValue("charge_min", 1, "Minimal possible charge")
    params.setValue("charge_max", 3, "Maximal possible charge")
    params.setValue("charge_span_max", 3)

    # maximum RT difference between any two features for grouping
    # maximum RT difference between between two co-features, after adduct shifts have been accounted for
    # (if you do not have any adduct shifts, this value should be equal to "retention_max_diff")
    params.setValue("retention_max_diff", 3.0)
    params.setValue("retention_max_diff_local", 3.0)

    # set updated paramters object
    mfd.setParameters(params)

    # result feature map: will store features with adduct information
    feature_map_MFD = oms.FeatureMap()
    # result consensus map: will store grouped features belonging to a charge group
    groups = oms.ConsensusMap()
    # result consensus map: will store paired features connected by an edge
    edges = oms.ConsensusMap()

    # compute adducts
    mfd.compute(fm, feature_map_MFD, groups, edges)

    # export feature map as pandas DataFrame and append adduct information
    ft = feature_map_MFD.get_df(export_peptide_identifications=False)
    ft["adduct"] = [f.getMetaValue("dc_charge_adducts") for f in feature_map_MFD]

    return ft, groups

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
    # extract all spectra from the MSExperiment instance
    specs={spec.getRT():{"mz":spec.get_peaks()[0],"int":spec.get_peaks()[1]} for spec in exp if spec.getMSLevel()==1}

    xics={}
    for i in ft.index:
        
        intsum_list=[]
        rtlist=[]
        # for each spectrum add the intensities of all m/z values in the defined range of the feature to the xic
        for rt,pk in specs.items():
            intsum_list.append(pk["int"][(ft.loc[i,"MZstart"]<=pk["mz"]) & (ft.loc[i,"MZend"]>=pk["mz"])].sum())
            rtlist.append(rt)

        xics[i]=pd.DataFrame({"rt":rtlist,"int":intsum_list})
    
    return xics,specs

def extract_xic_peakwindow(xic_dict:dict,ft:pd.DataFrame,window:float):
    """
    ## Description

    Extracts part of all xics defined by the location of their features peak-maximum and a given window.
    
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
        # Normalize peak
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

    # loops through all features correlated to the given activity peak by retention time
    for i in ft.loc[ft[f"corr_activity_peak{idx}"]>0].index:
        # get the xic of the current feature
        df=xic_dict[i]
        # check if the sampling rate of the feature xic is higher than that of the activity peak 
        if len(df)>=len(ap_df):
            # interpolate values in activity peak to match the samling rate of the feature xic
            interpRT=np.linspace(peak.RTstart,peak.RTend,len(df))
            interpInt=np.interp(interpRT,ap_df["RT"],ap_df[ydata_name])
            # calculate the pearson correlation of the peaks
            ft.loc[i,f"pearson_corr_peak{idx}"]=stats.pearsonr(df["int"],interpInt).statistic
        else:
            # interpolate values in the feature xic to match the sampling rate of the activity peak
            interpRT=np.linspace(df.rt.iloc[0],df.rt.iloc[-1],len(ap_df))
            interpInt=np.interp(interpRT,df.rt,df["int"])
            # Calculate the pearson correlation
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
    # extract only the current peak from the activity chromatogram
    cutap=act_df.loc[pk["start_idx"]:pk["end_idx"]].copy()
    # Normalize peak
    cutap[ydata_name]=cutap[ydata_name]/cutap[ydata_name].max()
    # determine the width of the peak in s
    width=np.abs(pk["RTend"]-pk["RTstart"])

    # extract the peak window
    cutxic=extract_xic_peakwindow(xic_dict,ft,width)
    
    peak_pearsoncorr(cutxic,ft,cutap,pk,idx,ydata_name)

def activity_annotation_features(ft:pd.DataFrame,aft:pd.DataFrame,act_df:pd.DataFrame,xic_dict:dict,rt_tolerance:float,rt_offset:float,ydata_name:str="norm_intensity"):
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
    |rt_offset|float|Offset of the activity chromatogram to the |
    |ydata_name|str|Name of the Column in act_df containing the y-axis information of the activity chromatogram|
    """
    for iat in aft.index:
        ft.loc[rt_tolerance>=np.abs(ft["RT"]-(aft.loc[iat,"RT"]+rt_offset)),f"corr_activity_peak{iat}"]=aft.loc[iat,"AUC"]
        ft[f"pearson_corr_peak{iat}"]=None

        peakshape_corr(xic_dict,ft,aft,act_df,iat,ydata_name)
