![Logo](assets/logo_uspotreader.png)

# User Guide for the MicrospotReader Web App

This document is directed at new users of the Web App implementation of the MicrospotReader Workflow. It will guide you through the individual steps required for annotation of LC-MS data with concomitantly determined bioactivity-values.

## Purpose

WIP

## Limitations

WIP

## Image Analysis

The Image Analysis page is responsible for extracting all required information from the performed optical bioactivity assay. It is important that the format of the recorded images is correct in order to yield a sucessful analysis.

### The Image

The image processing algorithm relies on a regular spacing of clearly visible spots or circular objects. Therefore images __must__ contain a regular grid of spots. As long as most spots within the image are clearly visible, artefacts from the inital detection can be corrected for. However, if noise within the image leads to a significant number of spots not being distinguishable from the background, entire rows or columns of spots might accidently be ignored. 

Ideally the images should be cropped such that they only contain the microPAD assay and nothing more. This should be done to avoid any artefacts during image processing. Aligning the microPAD assay with the image axes will also lead to more precise results and faster processing as the grid parameters (more on that soon) can be restricted more strongly.

Currently allowed image file formats include .tif, .png and .jpg as these have been tested so far. Allowed color spaces are RGB and grayscale. Since the image can be inverted in the Web App, it is not important whether the pixel intensity decreases or increases with the assays signal intensity.

An example of a reasonable image for analysis is shown here:

![Example Image for Analysis](assets/userguide/example_image.png)

### General Layout and Workflow

#### Settings

After uploading the image to be analyzed you will see the following setup: 

![Initial Layout Image Analysis](assets/userguide/layout1_image-analysis.png)

1. __The uploaded image:__ Version of the image to be used for densitometric analysis. Might be inverted if Nr. 2 is turned on.
2. __Invert grayscale image:__ Inverts the values of the grayscale image if turned on
3. __Index of the first spot:__ Corresponds to the top-left spot. Letter corresponds to row, number corresponds to column. Input is required!
4. __Index of the last spot:__ Corresponds to the bottom-right spot. Letter corresponds to row, number corresponds to column. Input is required!
5. __Enable halo detection:__ If turned on, will check each spot for an antimicrobial halo. Only enable, if a reporterstrain assay is performed in which halos are to be expected!
6. __Select Rows to be labeled as "control":__ Row-indices of entire rows used as negative control for normalization. Not required!
7. __Select Columns to be labeled as "control":__ Column-indices of entire columnes used as negative control for normalization. Not required!
8. __Advanced Settings:__ Contains tunable settings for all parameters needed during image analysis. Change with caution!
9. __Start Analysis Button:__ Starts the image analysis. The settings will disappear after pressing this button!
10. __Start New Analysis Button:__ Resets the current analysis so that a new one may be performed.

> Once all of the Settings have been set appropriately, the user can then press the ___Start Analysis!___ button in order to start the image processing workflow.

#### Results

After image analysis has been concluded the following set up will be shown:

![Layout of Results Image Analysis](assets/userguide/layout2_image-analysis.png)

1. __Remove false-positive Halos:__ Remove falsely detected halos from spots using their index.
2. __Tabs Displaying all Results:__ Tabs showing the relevant results from image analysis in tabular and visual form.
3. __Download Button:__ Download the resulting table as a .csv file.
4. __Store Data in the current session:__ For further processing of the data without the need to upload the resulting .csv files for every step, the data can be given a name and stored in the current session.
5. __Display of all stored image datasets:__ Displays the name of all datasets that have been stored in the session. The name of each dataset can be changed at any time. By selecting a dataset and clicking the button ___Delete Selection___, data can be removed from the session.

> Once all data has been downloaded and/or added to the session, one can continue to the next step: __Data Merging__

## Data Merging and Manipulation

## Annotation of .mzML-Files