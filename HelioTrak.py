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
from tqdm import tqdm
import multiprocessing
import time
from multiprocessing import Pool
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


###################################
##### HELPER FUNCTIONS ############
###################################

def housekeeping(datapath: str) -> None:

    """
    Ensures the existence and proper state of specific directories and their contents within a given data path.

    This function performs the following tasks:
    1. Checks if the directories "01-mask", "02-id", and "03-assoc" exist within the specified datapath.
       If none of these directories exist, it creates them.
    2. If the directories exist, it checks for files within them.
       - If all three directories contain the same number of files and are not empty, it prompts a warning
         message indicating that the directories are not empty and proceeds to delete all files in these directories.
       - If the number of files in "01-mask" and "02-id" do not match, it deletes all files in these two directories.
       - If the directories are empty, it prints a message indicating so.

    Parameters:
    datapath (str): The path where the directories "01-mask", "02-id", and "03-assoc" are located or should be created.

    Note:
    This function assumes the existence of a `color` class with BOLD, RED, GREEN, and END attributes for formatting output.

    Example:
    housekeeping("/path/to/root/data/folder/")

    PSA: This docstring has been written with the assistance of AI.
    """

    if not os.path.exists(datapath+"01-mask") and not os.path.exists(datapath+"02-id") and not os.path.exists(datapath+"03-assoc"):
        os.makedirs(datapath+"01-mask")
        os.makedirs(datapath+"02-id")
        os.makedirs(datapath+"03-assoc")
    else:
        files_mask = glob.glob(datapath+"01-mask/*")
        files_id = glob.glob(datapath+"02-id/*")
        files_assoc = glob.glob(datapath+"03-assoc/*")
        if len(files_mask) == len(files_id) == len(files_assoc) != 0:
            print(color.BOLD + color.RED + "WARNING: The directories are not empy. Deleting files..." + color.END)
            response = "y"
            if response == "y":
                for file in files_mask:
                    os.remove(file)
                for file in files_id:
                    os.remove(file)
                for file in files_assoc:
                    os.remove(file)
                print(color.BOLD + color.GREEN + "Files deleted successfully." + color.END)
            else:
                pass
                print(color.BOLD + color.GREEN + "Files not deleted." + color.END)
        elif len(files_mask) != len(files_id):
            print("The number of files in the directories 01-mask and 02-id do not match. Deleting all files.")
            for file in files_mask:
                os.remove(file)
            for file in files_id:
                os.remove(file)
        else:
            pass
            print("The directories are empty.")



def img_pre_pos(img: numpy.ndarray, l_thr: float) -> numpy.ndarray:
    img_pos = img.copy()
    img_pos[img_pos < 0] = 0
    img_pos = numpy.array(img_pos, dtype=numpy.float64)
    img_pos[img_pos < l_thr] = 0
    return img_pos

def img_pre_neg(img: numpy.ndarray, l_thr: float) -> numpy.ndarray:
    img_neg = img.copy()
    img_neg = -1*numpy.array(img_neg, dtype=numpy.float64)
    img_neg[img_neg < 0] = 0
    img_neg[img_neg < l_thr] = 0
    return img_neg


def watershed_routine(img: numpy.ndarray, min_dist: int, separation:bool = False) -> tuple[numpy.ndarray, numpy.ndarray]:
    if separation:
        distance = scipy.ndimage.distance_transform_edt(img)
        coords = skimage.feature.peak_local_max(distance, min_distance=min_dist)
        mask = numpy.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = scipy.ndimage.label(mask)
        labels_line = skimage.segmentation.watershed(-distance, markers, mask=img, compactness=0.001, watershed_line=True)
        return labels_line, coords
    elif not separation:
        coords = skimage.feature.peak_local_max(img, min_distance=min_dist)
        mask = numpy.zeros(img.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = scipy.ndimage.label(mask)
        labels = skimage.segmentation.watershed(-img, markers, mask=img, compactness=0.001)
        labels = numpy.array(labels, dtype=numpy.float64)
        return labels, coords
    elif type(separation) != bool:
        raise ValueError('separation must be a boolean')


###################################
##### MAIN FUNCTIONS ##############
###################################


def detection(img: numpy.ndarray, l_thr: float, min_distance:int,sign:bool="both", separation:bool=False) -> Union[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]:
    
    """
    Detects features in an image using a threshold and watershed algorithm based on the specified sign of features.

    This function processes an input image to detect features either of positive values, negative values, or both.
    The detection is based on thresholding and the watershed algorithm, which can be applied with or without separation.

    Parameters:
    img (numpy.ndarray): The input image to process.
    l_thr (int): The threshold value for detection.
    min_distance (int): The minimum distance between features for the watershed algorithm.
    sign (str, optional): Specifies the type of features to detect. Options are "both", "pos", or "neg".
                          Default is "both".
    separation (bool, optional): If True, applies separation in the watershed routine. Default is False.

    Returns:
    Union[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]:
        - If sign is "both": Returns a tuple of three elements:
            labels (numpy.ndarray): The combined labels of detected features.
            coords_pos (numpy.ndarray): Coordinates of positive features.
            coords_neg (numpy.ndarray): Coordinates of negative features.
        - If sign is "pos" or "neg": Returns a tuple of two elements:
            labels (numpy.ndarray): The labels of detected features.
            coords (numpy.ndarray): Coordinates of the detected features.

    Raises:
    ValueError: If sign is not "both", "pos", or "neg".

    Example:
    labels, coords_pos, coords_neg = detection(img, l_thr=50, min_distance=10, sign="both", separation=True)

    PSA: This docstring has been written with the assistance of AI.
    """
    
    img = numpy.array(img)
    if sign == "both":
        img_pos = img_pre_pos(img, l_thr)
        img_neg = img_pre_neg(img, l_thr)
        labels_pos,_ = watershed_routine(img_pos, min_distance, separation)
        labels_neg,_ = watershed_routine(img_neg, min_distance, separation)
        labels_neg = -1*labels_neg
        labels = labels_pos + labels_neg
        print(f"Number of clumps detected: {len(numpy.unique(labels))-1}")
        return labels
    elif sign == "pos":
        img_pos = img_pre_pos(img, l_thr)
        labels_pos,_ = watershed_routine(img_pos, min_distance, separation)
        return labels_pos
    elif sign == "neg":
        img_neg = img_pre_neg(img, l_thr)
        labels_neg,_ = watershed_routine(img_neg, min_distance, separation)
        return labels_neg
    else:
        raise ValueError('sign must be "both", "pos" or "neg"')


def identification(labels: numpy.ndarray, min_size: int) -> numpy.ndarray:

    """
    Identifies and filters clumps in the input label array based on a minimum size threshold.

    This function processes the input label array, retaining only those clumps (connected components) that meet 
    the specified minimum size. Clumps smaller than the minimum size are removed (set to zero).

    Parameters:
    labels (numpy.ndarray): The input array of labels representing different clumps.
    min_size (int): The minimum size (number of pixels) a clump must have to be retained.

    Returns:
    numpy.ndarray: The filtered label array with only clumps meeting the minimum size retained.

    Raises:
    ValueError: If no clumps survive the identification process.

    Note:
    Future versions may include a "verbose" option to print the number of clumps removed.

    Example:
    filtered_labels = identification(labels, min_size=50)

    PSA: This docstring has been written with the assistance of AI.
    """

    count = 0
    uid = numpy.unique(labels)
    original_number = len(uid)

    print(f"Number of clumps detected: {original_number-1}")

    for k in uid:
        sz = numpy.where(labels == k)
        if len(sz[0]) < min_size:
            labels = numpy.where(labels == k, 0, labels)
            count+=1

    num = original_number - count
    print(f"Number of clumps surviving the identification process: {num}")
    if num == 0:
        raise ValueError("No clumps survived the identification process")
    else: # In future versions, there should be a "verbose" option to print the number of clumps removed (num)
        print(f"Number of clumps surviving the identification process: {num}")
        pass

    return labels

def process_image(datapath: str, data: str, l_thr: int, min_distance: int, sign:bool="both", separation:bool=True, min_size:int=4) -> None:

    """
    Processes an astronomical image by detecting and identifying clumps within it, and saves the results.

    This function performs the following steps:
    1. Reads the input FITS image file.
    2. Detects clumps in the image using the specified detection parameters.
    3. Saves the detected clumps to the "01-mask" directory.
    4. Identifies and filters clumps based on the minimum size.
    5. Saves the filtered clumps to the "02-id" directory.

    Parameters:
    datapath (str): The path where the results will be saved.
    data (str): The path to the input FITS image file.
    l_thr (int): The threshold value for clump detection.
    min_distance (int): The minimum distance between detected clumps.
    sign (str, optional): Specifies the type of clumps to detect. Options are "both", "pos", or "neg".
                          Default is "both".
    separation (bool, optional): If True, applies separation in the detection routine. Default is True.
    min_size (int, optional): The minimum size (number of pixels) a clump must have to be retained. Default is 4.

    Returns:
    None

    Example:
    process_image("/path/to/data/", "image.fits", l_thr=50, min_distance=10, sign="both", separation=True, min_size=4)

    PSA: This docstring has been written with the assistance of AI.
    """

    image = astropy.io.fits.getdata(data, memmap=False)
    labels = detection(image, l_thr, min_distance, sign=sign, separation=separation)
    astropy.io.fits.writeto(datapath+f"01-mask/{data.split(os.sep)[-1]}", labels, overwrite=True)
    labels = identification(labels, min_size)
    astropy.io.fits.writeto(datapath+f"02-id/{data.split(os.sep)[-1]}", labels, overwrite=True)



def unique_id(id_data: str, datapath: str) -> None:

    """
    Assigns unique IDs to clumps in a list of FITS image files and saves the modified files.

    This function processes each FITS file in the provided list, replacing each unique non-zero clump identifier 
    with a globally unique ID. The modified images are saved in the "02-id" directory within the specified datapath.

    Parameters:
    id_data (list): A list of paths to the FITS files to be processed.
    datapath (str): The path where the modified files will be saved.

    Returns:
    None

    Example:
    unique_id(["image1.fits", "image2.fits"], "/path/to/data/")

    PSA: This docstring has been written with the assistance of AI.
    """

    u_id_p = 1
    u_id_n = -1
    for filename in id_data:
        img_n0 = astropy.io.fits.getdata(filename, memmap=False)
        # Extract unique non-zero values
        ids = numpy.unique(img_n0[img_n0 != 0])
        ids_p = ids[ids > 0]
        ids_n = ids[ids < 0]
        # Replace each unique value with its corresponding unique ID
        for i in ids_p:
            img_n0[img_n0 == i] = u_id_p
            u_id_p += 1
        for i in ids_n:
            img_n0[img_n0 == i] = u_id_n
            u_id_n -= 1
        # Write the modified data back to the file
        astropy.io.fits.writeto(os.path.join(datapath, "02-id", os.path.basename(filename)), img_n0, overwrite=True)
    print(f"Total number of unique IDs: {u_id_p+abs(u_id_n)-1}")



def array_row_intersection(a: numpy.ndarray,b:numpy.ndarray) -> numpy.ndarray:
   
    """
    Finds the intersection of rows between two 2D numpy arrays.

    This function identifies rows that are present in both input arrays `a` and `b` and returns the rows from `a`
    that are also in `b`. The function is optimized for performance using numpy operations.

    Parameters:
    a (numpy.ndarray): A 2D numpy array.
    b (numpy.ndarray): A 2D numpy array.

    Returns:
    numpy.ndarray: A 2D numpy array containing the rows from `a` that are also present in `b`.

    Note:
    This function is adapted from a solution provided by Vasilis Lemonidis on Stack Overflow.
    All credit goes to the original author. Source: https://stackoverflow.com/a/40600991

    Example:
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[3, 4], [7, 8]])
    result = array_row_intersection(a, b)
    # result will be array([[3, 4]])

    PSA: This docstring has been written with the assistance of AI.
    """

    tmp=numpy.prod(numpy.swapaxes(a[:,:,None],1,2)==b,axis=2)
    return a[numpy.sum(numpy.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]

def back_and_forth_matching_PARALLEL(fname1: str, fname2: str, round: int, datapath: str) -> None:

    """
    Performs parallel forward and backward matching of unique IDs between two FITS files.

    This function reads two FITS files, performs forward and backward matching of unique IDs
    based on the largest intersection of pixel coordinates, and saves the modified second FITS file.

    Parameters:
    fname1 (str): File path to the first FITS file.
    fname2 (str): File path to the second FITS file.
    round (int): Round number or identifier for the current processing round.
    datapath (str): Directory path where temporary and output files will be saved.

    Returns:
    None

    Note:
    This function assumes `array_row_intersection` function is defined elsewhere in your code.
    It uses tqdm for progress monitoring and assumes numpy arrays for file operations.

    Example:
    back_and_forth_matching_PARALLEL('file1.fits', 'file2.fits', 1, '/path/to/data/')

    PSA: This docstring has been written with the assistance of AI.
    """

    cube1 = astropy.io.fits.getdata(fname1, memmap=False)
    cube2 = astropy.io.fits.getdata(fname2, memmap=False)

    if len(numpy.shape(cube1)) == 2:
        file1 = cube1
    else: 
        file1 = cube1[-1, :, :]
    
    if len(numpy.shape(cube2)) == 2:
        file2 = cube2
    else:
        file2 = cube2[0, :, :]


    unique_id_1 = numpy.unique(file1)
    unique_id_1 = unique_id_1[unique_id_1 != 0]
    # create two empty 1D arrays to store the forward_1 and forward_2 matches
    forward_matches_1 = numpy.empty(0)
    forward_matches_2 = numpy.empty(0)
    for id_1 in unique_id_1:
        wh1 = numpy.where(file1 == id_1)
        set1 = numpy.stack((wh1[0], wh1[1])).T
        max_intersection_size = 0
        # create a mask of the element of the first image in the second image
        temp_mask = numpy.zeros(file2.shape)
        temp_mask[wh1] = 1
        temp_file2 = file2 * temp_mask
        unique_id_2 = numpy.unique(temp_file2)
        unique_id_2 = unique_id_2[unique_id_2 != 0]
        for id_2 in unique_id_2:
            wh2 = numpy.where(file2 == id_2)
            set2 = numpy.stack((wh2[0], wh2[1])).T
            temp_intersection_size = len(array_row_intersection(set1, set2))
            if temp_intersection_size > max_intersection_size:
                max_intersection_size = temp_intersection_size
                best_match_1 = id_1
                best_match_2 = id_2
        if max_intersection_size != 0:
            forward_matches_1 = numpy.append(forward_matches_1, best_match_1)
            forward_matches_2 = numpy.append(forward_matches_2, best_match_2)

    unique_id_2 = numpy.unique(file2)
    unique_id_2 = unique_id_2[unique_id_2 != 0]
    backward_matches_1 = numpy.empty(0)
    backward_matches_2 = numpy.empty(0)
    for id_2 in unique_id_2:
        wh2 = numpy.where(file2 == id_2)
        set2 = numpy.stack((wh2[0], wh2[1])).T
        max_intersection_size = 0
        # create a mask of the element of the first image in the second image
        temp_mask = numpy.zeros(file1.shape)
        temp_mask[wh2] = 1
        temp_file1 = file1 * temp_mask
        unique_id_1 = numpy.unique(temp_file1)
        unique_id_1 = unique_id_1[unique_id_1 != 0]
        for id_1 in unique_id_1:
            wh1 = numpy.where(file1 == id_1)
            set1 = numpy.stack((wh1[0], wh1[1])).T
            temp_intersection_size = len(array_row_intersection(set1, set2))
            if temp_intersection_size > max_intersection_size:
                max_intersection_size = temp_intersection_size
                best_match_1 = id_1
                best_match_2 = id_2
        if max_intersection_size != 0:
            backward_matches_1 = numpy.append(backward_matches_1, best_match_1)
            backward_matches_2 = numpy.append(backward_matches_2, best_match_2)

    # consider only the matches that are mutual
    mutual_matches_1 = numpy.empty(0)
    mutual_matches_2 = numpy.empty(0)
    for kk in range(len(forward_matches_1)):
        if forward_matches_1[kk] in backward_matches_1 and forward_matches_2[kk] in backward_matches_2:
            fwm1 = forward_matches_1[kk]
            fwm2 = forward_matches_2[kk]
            mutual_matches_1 = numpy.append(mutual_matches_1, fwm1)
            mutual_matches_2 = numpy.append(mutual_matches_2, fwm2)
    
    for idx in range(len(mutual_matches_1)):
        numpy.place(cube2, cube2 == mutual_matches_2[idx], mutual_matches_1[idx])
    
    # append vertically the frames in cube1 and cube2
    if len(numpy.shape(cube1)) == 2:
        cube1 = cube1.reshape(1, cube1.shape[0], cube1.shape[1])
    if len(numpy.shape(cube2)) == 2:
        cube2 = cube2.reshape(1, cube2.shape[0], cube2.shape[1])

    cube2 = numpy.concatenate((cube1, cube2), axis=0)

    astropy.io.fits.writeto(datapath+f"temp{round}/{fname1.split(os.sep)[-1]}", cube2, overwrite=True)
    print(color.YELLOW + f"Done with {fname1.split(os.sep)[-1]}, {fname2.split(os.sep)[-1]}" + color.END)


def associate(datapath: str) -> None:

    """
    Perform association of FITS files using parallel processing.

    This function processes FITS files in `datapath/02-id` directory, performing association
    using the `back_and_forth_matching_PARALLEL` function. It divides the data into subgroups,
    processes each subgroup in parallel using multiprocessing, and saves the associated results
    in `datapath/03-assoc` directory.

    Parameters:
    datapath (str): The base directory path containing the data.

    Returns:
    None

    Note:
    This function assumes the presence of necessary directories (`02-id` and `03-assoc`) and
    uses multiprocessing for parallelization. It also assumes the availability of `back_and_forth_matching_PARALLEL`
    function and `color` for colored console output.

    Example:
    associate("/path/to/data/")

    PSA: This docstring has been written with the assistance of AI.
    """

    print(color.RED + color.BOLD + "Starting association" + color.END)
    number_of_workers = os.cpu_count()
    id_data = sorted(glob.glob(datapath+"02-id/*.fits"))
    round = 0
    # make a folder named temp to store the results of the rounds of association
    os.makedirs(datapath+"temp0", exist_ok=True)
    # divide id_data in subgroups of 2 frames, in case of odd number of frames, the last frame is kept alone
    subgroups = [id_data[i:i+2] for i in range(0, len(id_data), 2)]
    if len(subgroups[-1]) == 1:
        img = astropy.io.fits.getdata(subgroups[-1][0], memmap=False)
        img = img.reshape(1, img.shape[0], img.shape[1])
        astropy.io.fits.writeto(datapath+f"temp{round}/{subgroups[-1][0].split(os.sep)[-1]}", img, overwrite=True)
        subgroups = subgroups[:-1]
    # parallelize the association by using the multiprocessing module on each subgroup
    print(color.RED + color.BOLD + "Starting the first round of association" + color.END)
    args = [(subgroup[0], subgroup[1], round, datapath) for subgroup in subgroups]
    pool = multiprocessing.Pool(processes=number_of_workers)
    results = pool.starmap(back_and_forth_matching_PARALLEL, args)
    pool.close()
    pool.join()

    max_iter = 10
    # subsequent rounds of association
    for round in range(1, max_iter):
        # load the data
        data = sorted(glob.glob(datapath+f"temp{round-1}/*.fits"))
        os.makedirs(datapath+f"temp{round}", exist_ok=True)
        if len(data) < 2:
            break
        # divide id_data in subgroups of 2 frames, in case of odd number of frames, the last frame is kept alone
        subgroups = [data[i:i+2] for i in range(0, len(data), 2)]
        if len(subgroups[-1]) == 1:
            img = astropy.io.fits.getdata(subgroups[-1][0], memmap=False)
            img = img.reshape(img.shape[0], img.shape[1], img.shape[2])
            astropy.io.fits.writeto(datapath+f"temp{round}/{subgroups[-1][0].split(os.sep)[-1]}", img, overwrite=True)
            subgroups = subgroups[:-1]
        
        print(color.RED + color.BOLD + f"Starting the {round+1} round of association" + color.END)
        args = [(subgroup[0], subgroup[1], round, datapath) for subgroup in subgroups]
        pool = multiprocessing.Pool(processes=number_of_workers)
        results = pool.starmap(back_and_forth_matching_PARALLEL, args)
        pool.close()
        pool.join()

    data = astropy.io.fits.getdata(sorted(glob.glob(datapath+f"temp{round-1}/*.fits"))[0], memmap=False)
    # export each frame of the data cube as a fits file in 03_assoc
    for i in range(data.shape[0]):
        astropy.io.fits.writeto(datapath+f"03-assoc/{i:04d}.fits", data[i, :, :], overwrite=True)

    print(color.GREEN + color.BOLD + "Finished association" + color.END)


def tabulation_parallel(files: str, filesB: str, dx: float, dt: float, cores: int, minliftime: int = 4) -> pandas.DataFrame:
    def process_file(j):
        file = files[j]
        src_img = astropy.io.fits.getdata(filesB[j], memmap=False)
        asc_img = astropy.io.fits.getdata(file, memmap=False)
        unique_ids = numpy.unique(asc_img)
        df_temp = pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "frame"])
        for i in unique_ids:
            if i == 0:
                continue
            mask = (asc_img == i)
            Bm = src_img * mask
            Area = mask.sum()
            Flux = Bm.sum() / Area
            X = ((mask * x_1) * Bm).sum() / Bm.sum()
            Y = ((mask * y_1) * Bm).sum() / Bm.sum()
            temp = pandas.DataFrame([[i, X, Y, Area, Flux, j]], columns=["label", "X", "Y", "Area", "Flux", "frame"])
            df_temp = pandas.concat([df_temp, temp], ignore_index=False)
        return df_temp

    df = pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "frame"])
    img = astropy.io.fits.getdata(filesB[0], memmap=False)
    size = numpy.shape(img)
    x_1, y_1 = numpy.meshgrid(numpy.arange(size[1]), numpy.arange(size[0]))

    with ProcessingPool(cores) as p:
        results = list(p.imap(process_file, range(len(files))))

    for result in results:
        df = pandas.concat([df, result], ignore_index=False)

    # Merge the common labels
    groups = df.groupby("label")

    area_tot = []
    flux_tot = []
    X_tot = []
    Y_tot = []
    label_tot = []
    frame_tot = []

    for name, group in groups:
        area_temp = group["Area"].values
        flux_temp = group["Flux"].values
        X_temp = group["X"].values
        Y_temp = group["Y"].values
        label_temp = group["label"].values
        frame_temp = group["frame"].values

        # Perform some sanity checks
        if len(area_temp) != len(flux_temp):
            raise ValueError("area and flux have different lengths")
        if len(area_temp) != len(X_temp):
            raise ValueError("area and X have different lengths")
        if len(area_temp) != len(Y_temp):
            raise ValueError("area and Y have different lengths")
        if len(area_temp) != len(label_temp):
            raise ValueError("area and label have different lengths")
        if len(area_temp) != len(frame_temp):
            raise ValueError("area and frame have different lengths")
        if len(numpy.unique(label_temp)) > 1:
            raise ValueError("More than one label in the group")
        if numpy.any(numpy.diff(frame_temp) != 1):
            raise ValueError("Frames are not consecutive")

        area_tot.append(area_temp)
        flux_tot.append(flux_temp)
        X_tot.append(X_temp)
        Y_tot.append(Y_temp)
        label_tot.append(label_temp)
        frame_tot.append(frame_temp)

    df_final = pandas.DataFrame(columns=["label", "Lifetime", "X", "Y", "Area", "Flux"])
    df_final["label"] = [x[0] for x in label_tot]
    df_final["Lifetime"] = [len(x) for x in frame_tot]
    df_final["X"] = X_tot
    df_final["Y"] = Y_tot
    df_final["Area"] = area_tot
    df_final["Flux"] = flux_tot
    df_final["Frames"] = frame_tot
    df_final = df_final[df_final["Lifetime"] >= minliftime]

    # Compute the velocities
    vxtot = []
    vytot = []
    stdvxtot = []
    stdvytot = []
    for j in range(len(df_final)):
        x = df_final["X"].iloc[j]
        y = df_final["Y"].iloc[j]
        x = numpy.array(x)
        y = numpy.array(y)
        vx = numpy.gradient(x) * dx / dt
        vy = numpy.gradient(y) * dx / dt
        vx = vx - numpy.mean(vx)
        vy = vy - numpy.mean(vy)
        stdx = numpy.std(vx)
        stdy = numpy.std(vy)
        vxtot.append(vx)
        vytot.append(vy)
        stdvxtot.append(stdx)
        stdvytot.append(stdy)
    df_final["Vx"] = vxtot
    df_final["Vy"] = vytot
    df_final["stdVx"] = stdvxtot
    df_final["stdVy"] = stdvytot

    df_final = df_final.reset_index(drop=True)

    return df_final


def tabulation(files: str, filesB: str,dx: float, dt: float, cores: int, minliftime:int=4) -> pandas.DataFrame:

    """
    Analyzes a series of FITS files and their corresponding background files to extract
    properties of labeled regions across frames, merge common labels, compute lifetime,
    and calculate velocities.

    Parameters:
    - files (list of str): List of paths to FITS files containing labeled regions.
    - filesB (list of str): List of paths to background FITS files corresponding to 'files'.
    - dx (float): Pixel size in the x-direction (spatial resolution).
    - dt (float): Time interval between frames (temporal resolution).
    - minliftime (int, optional): Minimum lifetime (number of frames a label persists) to consider. Default is 4.

    Returns:
    - pd.DataFrame: DataFrame containing tabulated data with columns:
      ['label', 'Lifetime', 'X', 'Y', 'Area', 'Flux', 'Frames', 'Vx', 'Vy', 'stdVx', 'stdVy'].

    Raises:
    - ValueError: If there are inconsistencies in the data, such as non-consecutive frames or multiple labels in a group.

    Notes:
    - The function assumes that 'files' and 'filesB' are aligned, i.e., each entry in 'files' corresponds to the
      same index in 'filesB' for background data.
    - 'dx' and 'dt' are used to compute velocities ('Vx', 'Vy') based on numerical differentiation of positions ('X', 'Y').
    PSA: This docstring has been written with the aid of AI.
    """

    df = pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "frame"])
    img = astropy.io.fits.getdata(filesB[0], memmap=False)
    size=numpy.shape(img)
    x_1, y_1 = numpy.meshgrid(numpy.arange(size[1]), numpy.arange(size[0])) 
    for j,file in tqdm(enumerate(files), desc="Tabulation"):
        src_img = astropy.io.fits.getdata(filesB[j], memmap=False)
        asc_img = astropy.io.fits.getdata(file, memmap=False)
        unique_ids = numpy.unique(asc_img)
        for i in unique_ids:
            if i == 0:
                continue
            mask=(asc_img == i)
            Bm=src_img*mask
            Area=mask.sum()
            Flux=Bm.sum()/Area
            X=((mask*x_1)*Bm).sum()/Bm.sum()
            Y=((mask*y_1)*Bm).sum()/Bm.sum()
            temp = pandas.DataFrame([[i, X, Y, Area, Flux, j]], columns=["label", "X", "Y", "Area", "Flux", "frame"])
            df = pandas.concat([df, temp], ignore_index=False)

    # Merge the common labels
    groups = df.groupby("label")

    area_tot = []
    flux_tot = []
    X_tot = []
    Y_tot = []
    label_tot = []
    frame_tot = []

    for name, group in tqdm(groups, desc="Merging common labels"):
        area_temp = group["Area"].values
        flux_temp = group["Flux"].values
        X_temp = group["X"].values
        Y_temp = group["Y"].values
        label_temp = group["label"].values
        frame_temp = group["frame"].values


        # Perform some sanity checks
        if len(area_temp) != len(flux_temp):
            raise ValueError("area and flux have different lengths")
        if len(area_temp) != len(X_temp):
            raise ValueError("area and X have different lengths")
        if len(area_temp) != len(Y_temp):
            raise ValueError("area and Y have different lengths")
        if len(area_temp) != len(label_temp):
            raise ValueError("area and label have different lengths")
        if len(area_temp) != len(frame_temp):
            raise ValueError("area and frame have different lengths")  
        if len(numpy.unique(label_temp)) > 1:
            raise ValueError("More than one label in the group")
        if numpy.any(numpy.diff(frame_temp) != 1):
            raise ValueError("Frames are not consecutive")
        
        area_tot.append(area_temp)
        flux_tot.append(flux_temp)
        X_tot.append(X_temp)
        Y_tot.append(Y_temp)
        label_tot.append(label_temp)
        frame_tot.append(frame_temp)



    df_final = pandas.DataFrame(columns=["label", "Lifetime", "X", "Y", "Area", "Flux"])
    df_final["label"] = [x[0] for x in label_tot]
    df_final["Lifetime"] = [len(x) for x in frame_tot]
    df_final["X"] = X_tot
    df_final["Y"] = Y_tot
    df_final["Area"] = area_tot
    df_final["Flux"] = flux_tot
    df_final["Frames"] = frame_tot
    df_final = df_final[df_final["Lifetime"] >= minliftime]

    # Compute the velocities
    vxtot = []
    vytot = []
    stdvxtot = []
    stdvytot = []
    for j in tqdm(range(len(df_final)), desc="Computing velocities"):
        x = df_final["X"].iloc[j]
        y = df_final["Y"].iloc[j]
        x = numpy.array(x)
        y = numpy.array(y)
        vx = numpy.gradient(x)*dx/dt
        vy = numpy.gradient(y)*dx/dt
        vx = vx-numpy.mean(vx)
        vy = vy-numpy.mean(vy)
        stdx = numpy.std(vx)
        stdy = numpy.std(vy)
        vxtot.append(vx)
        vytot.append(vy)
        stdvxtot.append(stdx)
        stdvytot.append(stdy)
    df_final["Vx"] = vxtot
    df_final["Vy"] = vytot
    df_final["stdVx"] = stdvxtot
    df_final["stdVy"] = stdvytot

    df_final = df_final.reset_index(drop=True)

    return df_final



###################################
##### WORKFLOW FUNCTION ###########
###################################

def track_all(datapath: str, cores: int, min_distance: int, l_thr: float, min_size: int, dx: float, dt: float, sign: str, separation: bool) -> None:

    """
    Executes a pipeline for feature detection, identification, association, tabulation, and data storage based on astronomical FITS files.

    Parameters:
    - datapath (str): Path to the main data directory.
    - cores (int): Number of CPU cores to utilize for parallel processing.
    - min_distance (int): Minimum distance between features for detection.
    - l_thr (int): Threshold value for feature detection.
    - min_size (int): Minimum size threshold for identified features.
    - dx (float): Pixel size in the x-direction (spatial resolution) for velocity computation.
    - dt (float): Time interval between frames (temporal resolution) for velocity computation.
    - sign (str): Sign convention for feature detection ('positive', 'negative', or 'both').
    - separation (bool): When true, it returns the separation lines of the watershed algorithm further separating features.

    Returns:
    - None: Outputs are saved as FITS files and a JSON file containing tabulated data.

    Raises:
    - FileNotFoundError: If there are issues with file paths or missing directories.

    Notes:
    - This function coordinates the detection, identification, association, tabulation, and storage of astronomical features
      across multiple FITS files in the specified 'datapath'.
    - The process involves multiple subprocesses, including cleaning up temporary files, feature detection, ID assignment,
      association of features across frames, tabulation of feature properties (such as position, area, and flux), and
      saving the resulting tabulated data as a JSON file.
    
    PSA: This docstring has been written with the aid of AI.
    """

    # Load the data
    data = sorted(glob.glob(datapath+"00-data/*.fits"))
    # Set the number of workers
    number_of_workers = numpy.min([len(data), cores])
    print(color.RED + color.BOLD + f"Number of cores used: {number_of_workers}" + color.END)
    start = time.time()
    # Clean up
    print(color.RED + color.BOLD + "Cleaning up" + color.END)
    housekeeping(datapath)
    # Start the detection and identification
    print(color.RED + color.BOLD + "Detecting features..." + color.END)
    print("Don't panic if it looks stuck...")

    with Pool(number_of_workers) as p:
        p.starmap(process_image, [(datapath, img, l_thr, min_distance, sign, separation, min_size) for img in data])
    # Assign unique IDs
    print(color.RED + color.BOLD + "Assigning unique IDs..." + color.END)
    id_data = sorted(glob.glob(datapath+"02-id/*.fits"))
    unique_id(id_data, datapath)
    print(color.GREEN + color.BOLD + "Feature detection step ended" + color.END)
    print(color.RED + color.BOLD + "Associating features..." + color.END)
    # Associate the detections
    associate(datapath)
    # delet all temp folders regardless of the files inside
    print(color.RED + color.BOLD + "Cleaning up" + color.END)
    os.system(f"rm -rf {datapath}temp*")
    # Start the tabulation
    print(color.RED + color.BOLD + "Starting tabulation" + color.END)
    asc_files = sorted(glob.glob(os.path.join(datapath,"03-assoc/*.fits")))
    src_files = sorted(glob.glob(os.path.join(datapath+"00-data/*.fits")))
    df = tabulation_parallel(asc_files, src_files, dx, dt, cores)
    # Save the dataframe
    df.to_json(os.path.join(datapath+"dataframe.json"))
    end = time.time()
    print(color.GREEN + color.BOLD + "Dataframe saved" + color.END)
    print(color.YELLOW + color.BOLD + f"Number of elements tracked: {len(df)}" + color.END)
    print(color.PURPLE + color.BOLD + f"Time elapsed: {end - start} seconds" + color.END)


###################################
##### UTILITY FUNCTIONS ###########
###################################


# These are all experimental functions and are not yet fully tested or optimized. Use with caution. They may be removed in future versions.

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


def freq_map(datapath, minlifetime=15):
    df = pandas.read_json(datapath+"dataframe.json")
    df = df[df["Lifetime"] > minlifetime]
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
        f, Pxx = scipy.signal.periodogram(numpy.array(df["Vx"].iloc[j]), fs=1/45, nfft=256, detrend="linear", scaling="density", )
        Pxx = Pxx/numpy.max(Pxx)
        peaks.append(f[numpy.argmax(Pxx)])

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
                    try:
                        peaks_img[y_+j, x_+k] = peaks[i]
                    except:
                        pass


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
    matplotlib.pyplot.show()
