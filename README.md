# SoFT: A Feature Tracking Suite for Solar Physics

Small-scale magnetic elements are vital in the energetic balance of the Sun’s atmosphere. These structures cover the entire solar surface and understanding their dynamics can address longstanding questions such as coronal heating and solar wind acceleration. **SoFT: Solar Feature Tracking** is a novel feature tracking routine built in Python, designed for reliable detection and fast associations.

### Detection and Identification: The Watershed Algorithm

The detection phase in SoFT involves:

1. **Threshold Masking**: Mask out pixels below a given threshold to reduce the impact of noise.
2. **Local Maxima Detection**: Identify peaks separated by a user-defined minimum distance.
3. **Euclidean Distance Transform (EDT)**: Compute the shortest distance from each non-zero pixel to the background.
4. **Watershed Segmentation**: Use local maxima as markers and segment the image based on the EDT gradient field.

### Association

Features are matched across frames:

1. **Forward Check**: Examine the overlap between feature M in frame n (M(n)) and all features in frame n+1 occupying the same pixels.
2. **Backward Check**: Verify the overlap between feature M in frame n+1 and features in frame n.
3. **Matching**: If M(n) and M(n+1) select each other, they are successfully matched.

To enable parallel processing, frames are paired and condensed into cubes. This reverse bisection condensation continues iteratively until one cube remains with all features properly associated.

### Tabulation

After association, the physical properties of magnetic structures are estimated and compiled:

- **Barycenters**: Calculated by averaging pixel coordinates weighted by intensity for sub-pixel accuracy.
- **Area**: Determined by counting pixels within the feature's contour.
- **Magnetic Flux**: Summed from pixel intensities.
- **Velocity**: Derived from the first-order derivative of barycenter positions.
- and many other


Further details regarding the SoFT tracking code and its performance can be found in [Berretti et al. 2024](https://doi.org/10.1051/0004-6361/202452665).

## Installation

Clone the repository and install the required dependencies:

```sh
git clone https://github.com/mib-unitn/SoFT.git
cd SoFT
pip install .
```
or

```sh
pip install solar_ft
```

## Usage

If you plan to use SoFT in your research or publications, please make sure to cite the corresponding paper: [Berretti et al. 2024](https://doi.org/10.1051/0004-6361/202452665).

```sh
import soft.soft as st
import os

#Set the path to the data
datapath = "path/to/data/"  # Path to the folder containing the "00-data" directory, which should include all the frames in single .fits files.
cores = os.cpu_count() # Sets the number of cores to be used. It will always be selected the minimum between the number of cores available and the number of frames in the data.


#Set the parameters for the detection and identification
l_thr =  #Intensity threshold[Gauss] (float)
m_size =  #Minimum size in pixels (int)
dx =  #Km (pixel size of the instrument) (float)
dt = #seconds (temporal cadence of the instrument) (float)
min_dist = # minimum required distance between two local maxima. (int)
sign = "both" # Can be "positive", "negative" or "both, defines the polarity of the features to be tracked (str)
separation = True  # If True, the detection method selected is "fine", if False, the detection method selected is "coarse". Check the paper for more details on the detection methods (bool)
verbose=False #If True, the code will print a more detailed output of the tracking process (bool)
doppler=False # If True, SoFT will also estimate the line-of-sight velocity within the detected features from separate dopplergram files in the 00b-data folder (bool)


st.track_all(datapath, cores, min_dist, l_thr, m_size, dx, dt, sign, separation, verbose, doppler)
```




<sub><sup><sub><sup><sub><sup><sub><sup><sub><sup><sub><sup><sub><sup><sub><sup> M. Berretti wishes to acknowledge that SoFT could also be interpreted as "So' Francesco Totti" and it's totally ok with it.</sup></sub></sup></sub></sup></sub></sup></sub></sup></sub></sup></sub></sup></sub></sup></sub>
