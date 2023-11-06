![image](assets/logo_uspotreader.png)

# MicroSpot Reader

Web-App for the detection and quantification of Spots on a microfluidics device for the determination of bioactivity of HPLC-fractions in parallel to an HPLC-MS experiment.


## Web-App

The Web-App is based on streamlit and currently runs on the streamlit cloud service:

[![Open Website!](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uspotreader.streamlit.app/)

### Local Installation:

1. Clone this repository
2. Open Windows Terminal and go to the main folder of the repository:

`cd <filepath>`

3. Create and activate a new python environment using Python 3.11 (example using anaconda):

`conda create --name microspotreader python=3.11`

`conda activate microspotreader`

4. Install all packages from the `requirements.txt` file:

`python -m pip install -r requirements.txt`

5. Start the App by running the following command:

`streamlit run MicrospotReader_App.py`


## Jupyter Notebooks

Additionally, this Repository contains Jupyter Notebooks in the `notebooks`-folder if you do not wish to use the Web-App:

- `microspot_reader.ipynb`: Detection and analysis of MicroSpots as well as antimicrobial halos within an image. Determination of bioactivity.

- `grid_concatenator.ipynb`: Concatenation of Spot-Lists of the same LC-MS run and correlation of MicroSpots with a retention time.

- `mzml_annotation.ipynb`: Annotation of MS1 spectra within a .mzML file with bioactivity at the corresponding retention time.