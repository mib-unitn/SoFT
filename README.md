# SoFT: A Feature Tracking Suite for Solar Physics

Small-scale magnetic elements are vital in the energetic balance of the Sun’s atmosphere. These structures cover the entire solar surface and understanding their dynamics can address longstanding questions such as coronal heating and solar wind acceleration.

While existing tracking codes are accessible, they often use outdated or licensed programming languages. **SoFT: Solar Feature Tracking** is a novel feature tracking routine built in Python, designed for reliable detection and fast associations.

### Detection and Identification: The Watershed Algorithm

The detection phase in SoFT involves:

1. **Threshold Masking**: Mask out pixels below a given threshold to reduce noise.
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

## Installation

Clone the repository and install the required dependencies:

```sh
git clone https://github.com/mib-unitn/SoFT.git
cd SoFT
pip install .
```

## Usage

```sh
import soft.soft as st

#Set the path to the data
datapath =  #Path to the data
# Set the number of cores to be used. It will always be selected the minimum between the number of cores available and the number of frames in the data.
cores = os.cpu_count()

#Set the parameters for the detection and identification
l_thr =  #Intensity [Gauss]
m_size =  #pixels
dx =  #Km
dt = #seconds
min_dist = #pixels
sign = # Can be "positive", "negative" or "both, defines the polarity of the features to be tracked
separation = # If True, the detection method selected is "fine", if False, the detection method selected is "coarse". Check the paper for more details on the detection methods
verbose=False #If True, the code will print a more detailed output of the tracking process


st.track_all(datapath, cores, min_dist, l_thr, m_size, dx, dt, sign, separation, verbose=False)
```




<sub><sup><sub><sup><sub><sup><sub><sup><sub><sup><sub><sup><sub><sup><sub><sup> M. Berretti wishes to acknowledge that SoFT could also be interpreted as "So' Francesco Totti" and it's totally ok with it.</sup></sub></sup></sub></sup></sub></sup></sub></sup></sub></sup></sub></sup></sub></sup></sub>