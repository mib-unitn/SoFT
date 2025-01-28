import numpy
import astropy.io.fits
import scipy
import skimage
import pandas
import os
import glob
from numba import jit
import matplotlib.pyplot
import matplotlib.animation
import tqdm 
import multiprocessing
import time
from typing import Union
from pathos.multiprocessing import ProcessingPool

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def trackvideo(images_files, cs_files,vmin, vmax, crop=False, filename="video", savepath=""):
    images = [astropy.io.fits.getdata(file, memmap=False) for file in images_files]
    cs = [astropy.io.fits.getdata(file, memmap=False) for file in cs_files]
    if crop:
        images = [image[int(image.shape[0]*0.4):int(image.shape[0]*0.6), int(image.shape[1]*0.4):int(image.shape[1]*0.6)] for image in images]
        cs = [c[int(c.shape[0]*0.4):int(c.shape[0]*0.6), int(c.shape[1]*0.4):int(c.shape[1]*0.6)] for c in cs]
    ims = []
    fig, ax= matplotlib.pyplot.subplots(1, 1, figsize=(10, 5))

    for j in range(len(cs)):
        im1 = ax.imshow(images[j], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        cs1 = ax.contour(cs[j], colors="red", origin='lower')

        ims.append([im1, cs1])


    ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save(savepath+f'{filename}.mp4', writer='ffmpeg', fps=5)


def extrusion(feature, df, asc_files, return_center=False):
    n = feature
    idx = df.index[df['label'] == n].tolist()[0]
    # get the corresponding frames from the dataframe
    frames = df.Frames[idx]
    x_center = df['X'][idx]
    y_center = df['Y'][idx]
    max_y = numpy.max(numpy.diff(x_center))
    max_x = numpy.max(numpy.diff(y_center))
    max_disp = int(numpy.max([max_x, max_y]))

    imgs = [astropy.io.fits.getdata(asc_files[i], memmap=False) for i in range(len(frames))]

    masks = [im == n for im in imgs]
    cropped_masks = [mask[int(y_center[0]-max_disp):int(y_center[0]+max_disp), int(x_center[0]-max_disp):int(x_center[0]+max_disp)] for mask in masks]
    x_center = numpy.array(x_center)
    y_center = numpy.array(y_center)
    stacked_masks = numpy.stack(cropped_masks, axis=2)
    z_center = numpy.array(frames)
    if return_center:
        x_center = x_center - (x_center[0]-max_disp)
        y_center = y_center - (y_center[0]-max_disp)
        return stacked_masks, x_center, y_center, z_center
    else:
        return stacked_masks, z_center
    

def follow_video(stacked_masks, savepath="", filename="follow_video"):
    ims = []
    fig, ax= matplotlib.pyplot.subplots(1, 1, figsize=(10, 5))
    for j in range(numpy.shape(stacked_masks)[2]):
        im1 = ax.imshow(stacked_masks[:,:,j], origin='lower', cmap='gray', vmin=0, vmax=1)
        ims.append([im1])

    ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save(savepath+f'{filename}.mp4', writer='ffmpeg', fps=5)


def freq_map(datapath, minlifetime=15, flag="x", title=None):
    df = pandas.read_json(datapath+"dataframe.json")
    df = df[df["Lifetime"] > minlifetime]
    df = df[df["stdVx"] < 10]
    df = df[df["stdVy"] < 10]
    df.reset_index(drop=True, inplace=True)
    img = astropy.io.fits.getdata(datapath+"00-data/0000.fits")
    cs = astropy.io.fits.getdata(datapath+"03-assoc/0000.fits")
    temp_img = numpy.copy(cs)
    lifetime = numpy.array(df["Lifetime"])

    peaks = []
    x = []
    y = []
    area = []


    for j in range(len(df)):
        x_ = numpy.array(df["X"].iloc[j]).mean()
        y_ = numpy.array(df["Y"].iloc[j]).mean()
        area_ = numpy.array(df["Area"].iloc[j]).mean()
        x.append(x_)
        y.append(y_)
        area.append(area_)
        if flag=="x":
            f, Pxx = scipy.signal.periodogram(numpy.array(df["Vx"].iloc[j]), fs=1/45, nfft=256, detrend="linear", scaling="density", )
            Pxx = Pxx/numpy.max(Pxx)
            peaks.append(f[numpy.argmax(Pxx)])
        elif flag=="y":
            f, Pyy = scipy.signal.periodogram(numpy.array(df["Vy"].iloc[j]), fs=1/45, nfft=256, detrend="linear", scaling="density", )
            Pyy = Pyy/numpy.max(Pyy)
            peaks.append(f[numpy.argmax(Pyy)])


    diameter = numpy.array(numpy.sqrt(numpy.array(area)/numpy.pi)*2).astype(int)
    peaks = numpy.array(peaks)

    peaks_img = numpy.zeros_like(temp_img).astype(numpy.float64)

    for i in range(len(peaks)):
        x_ = int(x[i])
        y_ = int(y[i])
        peaks_img[y_, x_] = peaks[i]
        for j in range(-diameter[i]//2, diameter[i]//2):
            for k in range(-diameter[i]//2, diameter[i]//2):
                if numpy.sqrt(j**2 + k**2) < diameter[i]//2:
                    peaks_img[y_+j, x_+k] = peaks[i]



    # smooth the peaks_img so that its all colored 
    from scipy.ndimage import gaussian_filter
    peaks_img = gaussian_filter(peaks_img, sigma=1)

    matplotlib.pyplot.figure(figsize=(10, 10))
    matplotlib.pyplot.imshow(img, cmap="gray", vmin=-100, vmax=100)
    # replace the values of the peaks_img with 0 to nan to make the background transparent
    peaks_img[peaks_img < 0.0007] = numpy.nan
    peaks_img = numpy.array(peaks_img)*1000

    matplotlib.pyplot.imshow(peaks_img, cmap="jet", alpha=0.99)
    matplotlib.pyplot.colorbar(label="Frequency (mHz)")
    if title != None:
        matplotlib.pyplot.savefig(f"{title}.png")

    

def amp_map(datapath, minlifetime=15, flag="x", title=None):
    df = pandas.read_json(datapath+"dataframe.json")
    df = df[df["Lifetime"] > minlifetime]
    df = df[df["stdVx"] < 10]
    df = df[df["stdVy"] < 10]
    df.reset_index(drop=True, inplace=True)
    img = astropy.io.fits.getdata(datapath+"00-data/0000.fits")
    cs = astropy.io.fits.getdata(datapath+"03-assoc/0000.fits")
    temp_img = numpy.copy(cs)
    lifetime = numpy.array(df["Lifetime"])

    peaks = []
    x = []
    y = []
    area = []


    for j in range(len(df)):
        x_ = numpy.array(df["X"].iloc[j]).mean()
        y_ = numpy.array(df["Y"].iloc[j]).mean()
        vx_ = numpy.array(df["Vx"].iloc[j]).std()
        vy_ = numpy.array(df["Vy"].iloc[j]).std()
        area_ = numpy.array(df["Area"].iloc[j]).mean()
        x.append(x_)
        y.append(y_)
        area.append(area_)
        if flag=="x":
            peaks.append(vx_)
        elif flag=="y":
            peaks.append(vy_)


    diameter = numpy.array(numpy.sqrt(numpy.array(area)/numpy.pi)*2).astype(int)
    peaks = numpy.array(peaks)
    # keep only values below 10


    peaks_img = numpy.zeros_like(temp_img).astype(numpy.float64)

    for i in range(len(peaks)):
        x_ = int(x[i])
        y_ = int(y[i])
        peaks_img[y_, x_] = peaks[i]
        for j in range(-diameter[i]//2, diameter[i]//2):
            for k in range(-diameter[i]//2, diameter[i]//2):
                if numpy.sqrt(j**2 + k**2) < diameter[i]//2:
                    peaks_img[y_+j, x_+k] = peaks[i]



    # smooth the peaks_img so that its all colored 
    from scipy.ndimage import gaussian_filter

    peaks_img = gaussian_filter(peaks_img, sigma=1)
    peaks_img[peaks_img < 0.001] = numpy.nan


    matplotlib.pyplot.figure(figsize=(10, 10))
    matplotlib.pyplot.imshow(img, cmap="gray", vmin=-100, vmax=100)
    # replace the values of the peaks_img with 0 to nan to make the background transparent


    matplotlib.pyplot.imshow(peaks_img, cmap="jet", alpha=0.99)
    matplotlib.pyplot.colorbar(label="Amplitude (km/s)")
    if title != None:
        matplotlib.pyplot.savefig(f"{title}.png")


def get_roi(img: numpy.ndarray, size: int = 20) -> tuple[numpy.ndarray, tuple[int, int]]:
    """
    Automatically identify the region of interest (ROI) in the image with the most values near zero.

    Parameters:
    img (numpy.ndarray): The input image.
    size (int): The size of the square ROI.

    Returns:
    tuple: A tuple containing the ROI and the bounding box coordinates (i, j).

    PSA: This docstring was written with the aid of AI.
    """
    roi = numpy.zeros((size, size))
    older_roi = numpy.ones((size, size)) * 1e9
    bbox = None
    for i in range(img.shape[0] - size):
        for j in range(img.shape[1] - size):
            roi = img[i:i + size, j:j + size]
            if numpy.mean(numpy.abs(roi)) < numpy.mean(numpy.abs(older_roi)):
                older_roi = roi
                bbox = (i, j)
    return older_roi, bbox