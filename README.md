![image](assets/Logo.png)

# MicroSpot Reader

Web-App for the detection and quantification of Spots on a microfluidics device for the determination of bioactivity of HPLC-fractions in parallel to an HPLC-MS experiment.


## Web-App

The Web-App is based on streamlit and currently runs on the streamlit cloud service:

[![Open Website!](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uspotreader.streamlit.app/)

### Local Installation:

1. Clone this repository
2. Open Windows Terminal and go to the main folder of the repository:

`cd <filepath>`

1. Create and activate a new conda environment:

`conda env create -f environment.yml`

`conda activate uspotreader`

4. Start the App by running the following command:

`streamlit run MicrospotReader_App.py`


## Jupyter Notebooks

Additionally, this Repository contains Jupyter Notebooks in the `notebooks`-folder if you do not wish to use the Web-App:

- `image_analysis.ipynb`: Detection and analysis of MicroSpots as well as antimicrobial halos within an image. Determination of bioactivity.

- `data_preparation.ipynb`: Concatenation of Spot-Lists of the same LC-MS run and correlation of MicroSpots with a retention time.

- `mzml_annotation.ipynb`: Annotation of MS1 spectra within a .mzML file with bioactivity at the corresponding retention time.

- `feature_finding.ipynb`: Feature detection and annotation with activity data from .csv file prepared with previous stepas and a .mzML file.