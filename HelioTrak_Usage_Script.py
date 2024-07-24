import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
import glob
import os
import HelioTrak as ht
import warnings
warnings.filterwarnings("ignore")

datapath = "path/to/data" # there should be a folder called 00-data with the fits files inside
data = sorted(glob.glob(os.path.join(datapath, "00-data/*.fits")))


# --- PARAMETERS --- #
#   PHI 106.16Km, 60s
#   IMaX 39.875Km, 20s
#   HMI 340.16Km, 45s

# get the number of cores
cores = os.cpu_count()

#Set the parameters for the detection and identification
l_thr = 12 #Intensity
m_size = 4 #pixels
dx = 340.16 #Km
dt = 45. #seconds
min_dist = 3 #pixels
sign ="both" # Can be "positive", "negative" or "both
separation = True # If True, the watershed algorithm will return the watershed line

ht.track_all(datapath, cores, min_dist, l_thr, m_size, dx, dt, sign, separation)
