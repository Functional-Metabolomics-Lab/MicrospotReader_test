import streamlit as st
import microspot_reader.streamlit as mst

mst.page_setup()

with st.sidebar:
    mst.datainfo()

st.image(r"assets/Logo.png")

st.markdown("### What is the MicroSpot-Reader WebApp used for?")

st.markdown("""
    The MicroSpot-Reader WebApp is a bioinformatic tool meant to assist researchers in the identification process of bioactive compounds in complex biological samples during Liquid Chromatography Mass Spectrometry (LC-MS) analysis.

    The tool employs an image processing algorithm to extract bioactivity data from optical biological or biochemical assays performed on LC fractions, particularly utilizing grid-based assay systems such as well plates or microfluidic devices. Subsequently, the tool constructs a bioactivity chromatogram, providing a comprehensive representation of bioactive peaks. This chromatogram is then meticulously correlated with features derived from untargeted metabolomics experiments. 

    By integrating the bioactivity information with mass spectrometric data, the tool facilitates the precise association of specific m/z-values with bioactive compounds. This synergistic approach aims not only to enhance the accuracy of bioactive compound identification but also provides valuable insights into the molecular basis of bioactivity within complex biological mixtures, thereby advancing our understanding of the functional aspects of metabolomes.
""")

st.markdown("### Input File Requirements")

st.markdown("""
            **Image File Requirements:**
            - File format: .png, .jpg, or .tif.
            - Color space: RGB or grayscale.
            - Image content: Circular wells or spots in a grid-based configuration.
            - Recommended: Crop image to only contain wells or spots.
            - If multiple assays are needed for one experiment: Capture one image per assay or split recorded image for independent analysis.
            - Avoid oversaturation to prevent distortion of peak shapes.
            
            **MS Data Requirements:**
            - Input: Centroided .mzML File of LC-MS run
            - Recommended: file from the same LC run as activity data
            - RT of compounds must be similar for activity data and MS data
            - Currently adduct detection is only possible for data measured in positive ion mode
            """)

st.markdown("### We value your Feedback!")

st.markdown("""
            We welcome your feedback and suggestions to improve the MicroSpot-Reader WebApp. Please feel free to create an issue on our GitHub repository to share your thoughts or report any issues you encounter. 
            Your input is invaluable in making the MicroSpot-Reader WebApp better for everyone.

            [Create an Issue on GitHub](https://github.com/Functional-Metabolomics-Lab/MicrospotReader/issues/new)
            """)

st.markdown("### Contribute and Follow Us")

st.markdown("""
- Interested in contributing? Check out our [GitHub personal page](https://github.com/sknesiron).
- For more about our work, visit our [lab's GitHub page](https://github.com/Functional-Metabolomics-Lab).
- Follow us on [Twitter](https://twitter.com/Functional-Metabolomics-Lab) for the latest updates.
""")