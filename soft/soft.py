import numpy
import astropy.io.fits
import scipy
import skimage
import pandas
import os
import glob
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



def track_all(datapath: str, cores: int, min_distance: int, l_thr: float, min_size: int, dx: float, dt: float, sign: str, separation: bool, verbose: bool = False, doppler: bool = False) -> None:

    """
    Executes a pipeline for feature detection, identification, association, tabulation, and data storage based on astronomical FITS files.

    Parameters:
    - datapath (str): Path to the main data directory.
    - cores (int): Number of CPU cores to utilize for parallel processing.
    - min_distance (int): Minimum distance between features for detection.
    - l_thr (float): Threshold value for feature detection.
    - min_size (int): Minimum size threshold for identified features.
    - dx (float): Pixel size in the x-direction (spatial resolution) for velocity computation.
    - dt (float): Time interval between frames (temporal resolution) for velocity computation.
    - sign (str): Sign convention for feature detection ('positive', 'negative', or 'both').
    - separation (bool): Separation threshold for feature detection.
    - verbose (bool, optional): If True, displays detailed progress information. Default is False.
    - doppler (bool, optional): If True, includes Doppler files for tabulation. Default is False.

    Returns:
    - None: Outputs are saved as FITS files and a JSON file containing tabulated data.
    """

    data = sorted(glob.glob(datapath + "00-data/*.fits"))
    number_of_workers = numpy.min([len(data), cores])
    if number_of_workers < 1:
        number_of_workers = 1
    print(color.RED + color.BOLD + f"Number of cores used: {number_of_workers}" + color.END)
    start = time.time()
    print(color.RED + color.BOLD + "Cleaning up" + color.END)
    housekeeping(datapath)
    print(color.RED + color.BOLD + "Detecting features..." + color.END)

    with multiprocessing.Pool(number_of_workers) as p:
        p.starmap(process_image, [(datapath, img, l_thr, min_distance, sign, separation, min_size, verbose) for img in data])

    print(color.RED + color.BOLD + "Assigning unique IDs..." + color.END)
    id_data = sorted(glob.glob(datapath + "02-id/*.fits"))
    unique_id(id_data, datapath, verbose)
    print(color.GREEN + color.BOLD + "Feature detection step ended" + color.END)
    print(color.RED + color.BOLD + "Associating features..." + color.END)
    associate(datapath, verbose, number_of_workers)
    print(color.RED + color.BOLD + "Cleaning up" + color.END)
    os.system(f"rm -rf {datapath}temp*")
    print(color.RED + color.BOLD + "Starting tabulation" + color.END)
    asc_files = sorted(glob.glob(os.path.join(datapath, "03-assoc/*.fits")))
    src_files = sorted(glob.glob(os.path.join(datapath + "00-data/*.fits")))
    if len(asc_files) == 0 or len(src_files) == 0:
        print(color.RED + color.BOLD + "No association or source files found for tabulation." + color.END)
        return

    if doppler:
        doppler_files = sorted(glob.glob(os.path.join(datapath + "00b-doppler/*.fits")))
        if len(doppler_files) == 0:
            raise FileNotFoundError("No Doppler files found")
        df = tabulation_parallel_doppler(asc_files, doppler_files, src_files, dx, dt, cores)
    else:
        df = tabulation_parallel(asc_files, src_files, dx, dt, cores)

    df.to_json(os.path.join(datapath + "dataframe.json"))
    end = time.time()
    print(color.GREEN + color.BOLD + "Dataframe saved" + color.END)
    print(color.YELLOW + color.BOLD + f"Number of elements tracked: {len(df)}" + color.END)
    print(color.PURPLE + color.BOLD + f"Time elapsed: {end - start} seconds" + color.END)



###################################
##### HELPER FUNCTIONS ############
###################################

def housekeeping(datapath: str) -> None:
    """
    Ensures the existence and proper state of specific directories and their
    contents within a given data path.
    """
    if not os.path.exists(datapath + "01-mask") and not os.path.exists(datapath + "02-id") and not os.path.exists(datapath + "03-assoc"):
        os.makedirs(datapath + "01-mask")
        os.makedirs(datapath + "02-id")
        os.makedirs(datapath + "03-assoc")
    else:
        files_mask = glob.glob(datapath + "01-mask/*")
        files_id = glob.glob(datapath + "02-id/*")
        files_assoc = glob.glob(datapath + "03-assoc/*")
        if len(files_mask) == len(files_id) == len(files_assoc) != 0:
            print(color.BOLD + color.RED + "WARNING: The directories are not empty. Deleting files..." + color.END)
            for file in files_mask:
                os.remove(file)
            for file in files_id:
                os.remove(file)
            for file in files_assoc:
                os.remove(file)
            print(color.BOLD + color.GREEN + "Files deleted successfully." + color.END)
        elif len(files_mask) != len(files_id):
            print("The number of files in the directories 01-mask and 02-id do not match. Deleting all files.")
            for file in files_mask:
                os.remove(file)
            for file in files_id:
                os.remove(file)
        else:
            print("The directories are empty.")


def img_pre_pos(img: numpy.ndarray, thr: float) -> numpy.ndarray:
    img_pos = img.copy()
    img_pos[img_pos < 0] = 0
    img_pos = numpy.array(img_pos, dtype=numpy.float64)
    img_pos[img_pos < thr] = 0
    return img_pos

def img_pre_neg(img: numpy.ndarray, l_thr: float) -> numpy.ndarray:
    img_neg = img.copy()
    img_neg = -1 * numpy.array(img_neg, dtype=numpy.float64)
    img_neg[img_neg < 0] = 0
    img_neg[img_neg < l_thr] = 0
    return img_neg


def watershed_routine(img: numpy.ndarray, l_thr: float, min_dist: int, sign: str, separation: bool = False) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Segments an image using a watershed algorithm.

    When separation=False, exactly one seed is placed per connected component
    at the distance-transform maximum — guaranteeing every blob above l_thr
    gets a label regardless of its peak intensity.

    When separation=True, multiple seeds per component are allowed, spaced at
    least min_dist apart (greedy selection on the distance transform), so that
    touching features can be split without losing weaker blobs.

    Parameters
    ----------
    img : ndarray
        Input 2D image.
    l_thr : float
        Lower intensity threshold; pixels below this are treated as background.
    min_dist : int
        Minimum distance between seeds (only enforced when separation=True).
    sign : {"pos", "neg"}
        Whether to detect positive or negative features.
    separation : bool
        If True, uses watershed lines and compactness to split touching blobs.

    Returns
    -------
    labels_line : ndarray
        Label array (background = 0).
    coords : ndarray, shape (N, 2)
        Seed coordinates used for the watershed.
    """
    if sign == "neg":
        img_low = img_pre_neg(img, l_thr)
    elif sign == "pos":
        img_low = img_pre_pos(img, l_thr)
    else:
        raise ValueError('sign must be "neg" or "pos"')

    distance = scipy.ndimage.distance_transform_edt(img_low)

    low_labels = skimage.measure.label(img_low > 0)
    all_components = set(numpy.unique(low_labels)) - {0}

    if not all_components:
        return numpy.zeros_like(img_low, dtype=int), numpy.empty((0, 2))

    if not separation:
        # One seed per component at its distance-transform maximum.
        coords = []
        for comp in all_components:
            comp_points = numpy.argwhere(low_labels == comp)
            vals = distance[comp_points[:, 0], comp_points[:, 1]]
            coords.append(comp_points[numpy.argmax(vals)].astype(float))
        coords = numpy.array(coords)

    else:
        # Multiple seeds per component, greedily spaced by min_dist.
        # This allows touching blobs to be separated while still guaranteeing
        # every component gets at least one seed.
        coords = []
        for comp in all_components:
            comp_mask = low_labels == comp
            comp_points = numpy.argwhere(comp_mask)
            vals = distance[comp_points[:, 0], comp_points[:, 1]]
            order = numpy.argsort(-vals)  # descending by distance value

            seeds = []
            for idx in order:
                pt = comp_points[idx].astype(float)
                if len(seeds) == 0:
                    # Always place at least one seed (the global maximum)
                    seeds.append(pt)
                else:
                    dists = numpy.sqrt(numpy.sum((numpy.array(seeds) - pt) ** 2, axis=1))
                    if numpy.all(dists >= min_dist):
                        seeds.append(pt)
            coords.extend(seeds)

        coords = numpy.array(coords) if coords else numpy.empty((0, 2))

    if coords.size == 0:
        return numpy.zeros_like(img_low, dtype=int), numpy.empty((0, 2))

    seed_mask = numpy.zeros(distance.shape, dtype=bool)
    seed_mask[tuple(coords.astype(int).T)] = True
    markers, _ = scipy.ndimage.label(seed_mask)
    labels_line = skimage.segmentation.watershed(
        -distance, markers, mask=img_low > 0,
        compactness=10 if separation else 0,
        watershed_line=separation
    )
    return labels_line, coords


###################################
##### MAIN FUNCTIONS ##############
###################################


def detection(img: numpy.ndarray, l_thr: float, min_distance: int, sign: str = "both", separation: bool = False, verbose: bool = False) -> numpy.ndarray:
    """
    Detects features in an image using a threshold and watershed algorithm.

    Parameters
    ----------
    img : ndarray
        Input image.
    l_thr : float
        Intensity threshold for detection. Pixels below this are ignored.
    min_distance : int
        Minimum distance between seeds (used only when separation=True).
    sign : {"both", "pos", "neg"}
        Whether to detect positive, negative, or both types of features.
    separation : bool
        If True, applies watershed separation to split touching features.
    verbose : bool
        If True, prints the number of detected clumps.

    Returns
    -------
    labels : ndarray
        Label array. Positive labels = positive features, negative = negative.
    """
    img = numpy.array(img)
    if sign == "both":
        labels_pos, _ = watershed_routine(img, l_thr, min_distance, "pos", separation)
        labels_neg, _ = watershed_routine(img, l_thr, min_distance, "neg", separation)
        labels_neg = -1 * labels_neg
        labels = labels_pos + labels_neg
        if verbose:
            print(f"Number of clumps detected: {len(numpy.unique(labels)) - 1}")
        return labels
    elif sign == "pos":
        labels_pos, _ = watershed_routine(img, l_thr, min_distance, "pos", separation)
        return labels_pos
    elif sign == "neg":
        labels_neg, _ = watershed_routine(img, l_thr, min_distance, "neg", separation)
        return labels_neg
    else:
        raise ValueError('sign must be "both", "pos" or "neg"')


def identification(labels: numpy.ndarray, min_size: int, verbose: bool = False) -> numpy.ndarray:
    """
    Filters clumps in the label array, keeping only those >= min_size pixels.

    Parameters
    ----------
    labels : ndarray
        Label array from detection().
    min_size : int
        Minimum number of pixels a clump must have to survive.
    verbose : bool
        If True, prints clump counts.

    Returns
    -------
    labels : ndarray
        Filtered label array.
    """
    count = 0
    uid = numpy.unique(labels)
    original_number = len(uid)
    if verbose:
        print(f"Number of clumps detected: {original_number - 1}")

    for k in tqdm.tqdm(uid, leave=False):
        sz = numpy.where(labels == k)
        if len(sz[0]) < min_size:
            labels = numpy.where(labels == k, 0, labels)
            count += 1

    num = original_number - count
    if num == 0:
        raise ValueError("No clumps survived the identification process")
    if verbose:
        print(f"Number of clumps surviving the identification process: {num}")

    return labels


def process_image(datapath: str, data: str, l_thr: float, min_distance: int, sign: str = "both", separation: bool = True, min_size: int = 4, verbose: bool = False) -> None:
    """
    Processes a single FITS image: detects clumps, filters by size, and saves results.

    Parameters
    ----------
    datapath : str
        Root output directory.
    data : str
        Path to the input FITS file.
    l_thr : float
        Intensity threshold for detection.
    min_distance : int
        Minimum seed spacing (separation=True only).
    sign : {"both", "pos", "neg"}
        Polarity of features to detect.
    separation : bool
        Whether to apply watershed separation.
    min_size : int
        Minimum clump size in pixels.
    verbose : bool
        Verbose output.
    """
    image = astropy.io.fits.getdata(data, memmap=False)
    labels = detection(image, l_thr, min_distance, sign=sign, separation=separation, verbose=verbose)
    astropy.io.fits.writeto(datapath + f"01-mask/{data.split(os.sep)[-1]}", labels, overwrite=True)
    labels = identification(labels, min_size, verbose=verbose)
    astropy.io.fits.writeto(datapath + f"02-id/{data.split(os.sep)[-1]}", labels, overwrite=True)


def unique_id(id_data: str, datapath: str, verbose: bool = False) -> None:
    """
    Assigns globally unique IDs to clumps across all frames.
    """
    u_id_p = 1
    u_id_n = -1
    for filename in tqdm.tqdm(id_data):
        img_n0 = astropy.io.fits.getdata(filename, memmap=False)
        ids = numpy.unique(img_n0[img_n0 != 0])
        ids_p = ids[ids > 0]
        ids_n = ids[ids < 0]
        for i in ids_p:
            img_n0[img_n0 == i] = u_id_p
            u_id_p += 1
        for i in ids_n:
            img_n0[img_n0 == i] = u_id_n
            u_id_n -= 1
        astropy.io.fits.writeto(os.path.join(datapath, "02-id", os.path.basename(filename)), img_n0, overwrite=True)
    if verbose:
        print(f"Total number of unique IDs: {u_id_p + abs(u_id_n) - 1}")


def array_row_intersection(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    """
    Returns rows of `a` that also appear in `b`.
    Adapted from https://stackoverflow.com/a/40600991 (Vasilis Lemonidis).
    """
    tmp = numpy.prod(numpy.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return a[numpy.sum(numpy.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)]


def back_and_forth_matching_PARALLEL(fname1: str, fname2: str, round: int, datapath: str, verbose: bool = False) -> None:
    """
    Forward/backward matching of unique IDs between two FITS files,
    keeping only mutually consistent matches.
    """
    cube1 = astropy.io.fits.getdata(fname1, memmap=False)
    cube2 = astropy.io.fits.getdata(fname2, memmap=False)

    file1 = cube1[-1] if cube1.ndim > 2 else cube1
    file2 = cube2[0] if cube2.ndim > 2 else cube2

    unique_id_1 = numpy.unique(file1)
    unique_id_1 = unique_id_1[unique_id_1 != 0]
    forward_matches_1 = numpy.empty(0)
    forward_matches_2 = numpy.empty(0)
    for id_1 in tqdm.tqdm(unique_id_1, leave=False, desc="Forward matching"):
        try:
            wh1 = numpy.where(file1 == id_1)
            set1 = numpy.stack((wh1[0], wh1[1])).T
        except Exception:
            raise ValueError(f"Error in forward matching for id_1: {id_1}, frame {fname1.split(os.sep)[-1]}, round {round}")
        max_intersection_size = 0
        temp_mask = numpy.where(file1 == id_1, 1, 0)
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
    for id_2 in tqdm.tqdm(unique_id_2, leave=False, desc="Backward matching"):
        try:
            wh2 = numpy.where(file2 == id_2)
            set2 = numpy.stack((wh2[0], wh2[1])).T
        except Exception:
            raise ValueError(f"Error in backward matching for id_2: {id_2}, frame {fname2.split(os.sep)[-1]}, round {round}")
        max_intersection_size = 0
        temp_mask = numpy.where(file2 == id_2, 1, 0)
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

    mutual_matches_1 = numpy.empty(0)
    mutual_matches_2 = numpy.empty(0)
    for kk in tqdm.tqdm(range(len(forward_matches_1)), leave=False, desc="Mutual matching"):
        if forward_matches_1[kk] in backward_matches_1 and forward_matches_2[kk] in backward_matches_2:
            mutual_matches_1 = numpy.append(mutual_matches_1, forward_matches_1[kk])
            mutual_matches_2 = numpy.append(mutual_matches_2, forward_matches_2[kk])

    for idx in tqdm.tqdm(range(len(mutual_matches_1)), leave=False, desc="Replacing"):
        numpy.place(cube2, cube2 == mutual_matches_2[idx], mutual_matches_1[idx])

    if len(numpy.shape(cube1)) == 2:
        cube1 = cube1.reshape(1, cube1.shape[0], cube1.shape[1])
    if len(numpy.shape(cube2)) == 2:
        cube2 = cube2.reshape(1, cube2.shape[0], cube2.shape[1])

    cube2 = numpy.concatenate((cube1, cube2), axis=0)
    astropy.io.fits.writeto(datapath + f"temp{round}/{fname1.split(os.sep)[-1]}", cube2, overwrite=True)
    if verbose:
        print(color.YELLOW + f"Done with {fname1.split(os.sep)[-1]}, {fname2.split(os.sep)[-1]}" + color.END)


def associate(datapath: str, verbose: bool = False, number_of_workers: int = None) -> None:
    """
    Performs iterative pairwise association of all frames using parallel processing.
    """
    id_data = sorted(glob.glob(datapath + "02-id/*.fits"))
    round = 0
    os.makedirs(datapath + "temp0", exist_ok=True)
    subgroups = [id_data[i:i + 2] for i in range(0, len(id_data), 2)]
    if len(subgroups) > 0 and len(subgroups[-1]) == 1:
        img = astropy.io.fits.getdata(subgroups[-1][0], memmap=False)
        img = img.reshape(1, img.shape[0], img.shape[1])
        astropy.io.fits.writeto(datapath + f"temp{round}/{subgroups[-1][0].split(os.sep)[-1]}", img, overwrite=True)
        subgroups = subgroups[:-1]
    print(color.RED + color.BOLD + "Starting the first round of association" + color.END)
    args = [(subgroup[0], subgroup[1], round, datapath, verbose) for subgroup in subgroups]
    pool = multiprocessing.Pool(processes=number_of_workers)
    pool.starmap(back_and_forth_matching_PARALLEL, args)
    pool.close()
    pool.join()

    max_iter = 10
    for round in range(1, max_iter):
        data = sorted(glob.glob(datapath + f"temp{round - 1}/*.fits"))
        os.makedirs(datapath + f"temp{round}", exist_ok=True)
        if len(data) < 2:
            break
        subgroups = [data[i:i + 2] for i in range(0, len(data), 2)]
        if len(subgroups) > 0 and len(subgroups[-1]) == 1:
            img = astropy.io.fits.getdata(subgroups[-1][0], memmap=False)
            img = img.reshape(img.shape[0], img.shape[1], img.shape[2])
            astropy.io.fits.writeto(datapath + f"temp{round}/{subgroups[-1][0].split(os.sep)[-1]}", img, overwrite=True)
            subgroups = subgroups[:-1]
        print(color.RED + color.BOLD + f"Starting the {round + 1} round of association" + color.END)
        args = [(subgroup[0], subgroup[1], round, datapath, verbose) for subgroup in subgroups]
        pool = multiprocessing.Pool(processes=number_of_workers)
        pool.starmap(back_and_forth_matching_PARALLEL, args)
        pool.close()
        pool.join()

    final_files = sorted(glob.glob(datapath + f"temp{round - 1}/*.fits"))
    if not final_files:
        print(color.RED + color.BOLD + "No associated files found." + color.END)
        return

    data = astropy.io.fits.getdata(final_files[0], memmap=False)
    for i in range(data.shape[0]):
        astropy.io.fits.writeto(datapath + f"03-assoc/{i:04d}.fits", data[i, :, :], overwrite=True)
    print(color.GREEN + color.BOLD + "Finished association" + color.END)


def tabulation_parallel(files: list, filesB: list, dx: float, dt: float, cores: int, minliftime: int = 4) -> pandas.DataFrame:
    """
    Process segmentation maps and source FITS files in parallel.
    Extracts blob properties, tracks across frames, and computes velocities.
    """
    img = astropy.io.fits.getdata(filesB[0], memmap=False)
    size = numpy.shape(img)
    x_1, y_1 = numpy.meshgrid(numpy.arange(size[1]), numpy.arange(size[0]))

    def process_file(j):
        src_img = astropy.io.fits.getdata(filesB[j], memmap=False)
        asc_img = astropy.io.fits.getdata(files[j], memmap=False)
        unique_ids = numpy.unique(asc_img)
        records = []
        for i in tqdm.tqdm(unique_ids, leave=False, desc=f"Frame {j}"):
            if i == 0:
                continue
            mask = (asc_img == i)
            Bm = src_img * mask
            Area = mask.sum()
            if Area == 0 or Bm.sum() == 0:
                continue
            Flux = Bm.sum() / Area
            X = ((mask * x_1) * Bm).sum() / Bm.sum()
            Y = ((mask * y_1) * Bm).sum() / Bm.sum()
            r = numpy.sqrt(Area / numpy.pi)
            circle = ((x_1 - X) ** 2 + (y_1 - Y) ** 2 < r ** 2).astype(int) * mask
            ecc = circle.sum() / Area if Area > 0 else numpy.nan
            records.append({"label": i, "X": X, "Y": Y, "Area": Area, "Flux": Flux, "frame": j, "ecc": ecc})
        return pandas.DataFrame.from_records(records)

    with ProcessingPool(cores) as p:
        results = list(p.imap(process_file, range(len(files))))

    df = pandas.concat(results, ignore_index=True)
    groups = df.groupby("label")

    area_tot, flux_tot, X_tot, Y_tot = [], [], [], []
    label_tot, frame_tot, ecc_tot = [], [], []

    for name, group in groups:
        area_temp = group["Area"].values
        flux_temp = group["Flux"].values
        X_temp = group["X"].values
        Y_temp = group["Y"].values
        label_temp = group["label"].values
        frame_temp = group["frame"].values
        ecc_temp = group["ecc"].values

        if not (len(area_temp) == len(flux_temp) == len(X_temp) == len(Y_temp) == len(label_temp) == len(frame_temp)):
            raise ValueError(f"Inconsistent lengths in group {name}")
        if len(numpy.unique(label_temp)) > 1:
            raise ValueError(f"More than one label in group {name}")
        if numpy.any(numpy.diff(frame_temp) != 1):
            raise ValueError(f"Frames are not consecutive for label {name}")

        area_tot.append(area_temp)
        flux_tot.append(flux_temp)
        X_tot.append(X_temp)
        Y_tot.append(Y_temp)
        label_tot.append(label_temp)
        frame_tot.append(frame_temp)
        ecc_tot.append(ecc_temp)

    df_final = pandas.DataFrame({
        "label": [x[0] for x in label_tot],
        "Lifetime": [len(x) for x in frame_tot],
        "X": X_tot, "Y": Y_tot,
        "Area": area_tot, "Flux": flux_tot,
        "Frames": frame_tot, "ecc": ecc_tot
    })
    df_final = df_final[df_final["Lifetime"] >= minliftime]

    vxtot, vytot, stdvxtot, stdvytot = [], [], [], []
    for j in range(len(df_final)):
        x = numpy.array(df_final["X"].iloc[j])
        y = numpy.array(df_final["Y"].iloc[j])
        vx = numpy.gradient(x) * dx / dt
        vy = numpy.gradient(y) * dx / dt
        vxtot.append(vx)
        vytot.append(vy)
        stdvxtot.append(numpy.std(vx))
        stdvytot.append(numpy.std(vy))

    df_final["Vx"] = vxtot
    df_final["Vy"] = vytot
    df_final["stdVx"] = stdvxtot
    df_final["stdVy"] = stdvytot
    df_final = df_final.reset_index(drop=True)
    return df_final


def tabulation_parallel_doppler(files: str, filesD: str, filesB: str, dx: float, dt: float, cores: int, minliftime: int = 4) -> pandas.DataFrame:

    img = astropy.io.fits.getdata(filesB[0], memmap=False)
    size = numpy.shape(img)
    x_1, y_1 = numpy.meshgrid(numpy.arange(size[1]), numpy.arange(size[0]))

    def process_file(j):
        src_img = astropy.io.fits.getdata(filesB[j], memmap=False)
        asc_img = astropy.io.fits.getdata(files[j], memmap=False)
        alt_img = astropy.io.fits.getdata(filesD[j], memmap=False)
        unique_ids = numpy.unique(asc_img)
        df_temp = pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "LOS_V", "frame", "ecc"])
        for i in unique_ids:
            if i == 0:
                continue
            mask = (asc_img == i)
            Bm = src_img * mask
            LosV_s = alt_img * mask
            LosV_s[LosV_s == 0] = numpy.nan
            Area = mask.sum()
            Flux = Bm.sum() / Area
            X = ((mask * x_1) * Bm).sum() / Bm.sum()
            Y = ((mask * y_1) * Bm).sum() / Bm.sum()
            r = numpy.sqrt(Area / numpy.pi)
            circle = ((x_1 - X) ** 2 + (y_1 - Y) ** 2 < r ** 2).astype(int) * mask
            ecc = circle.sum() / Area
            temp = pandas.DataFrame([[i, X, Y, Area, Flux, LosV_s, j, ecc]], columns=["label", "X", "Y", "Area", "Flux", "LOS_V", "frame", "ecc"])
            df_temp = pandas.concat([df_temp, temp], ignore_index=False)
        return df_temp

    with ProcessingPool(cores) as p:
        results = list(p.imap(process_file, range(len(files))))

    df = pandas.concat(results, ignore_index=True)
    groups = df.groupby("label")

    area_tot, flux_tot, losv_tot, X_tot, Y_tot, label_tot, frame_tot, ecc_tot = [], [], [], [], [], [], [], []

    for name, group in tqdm.tqdm(groups, desc="Merging common labels"):
        area_temp = group["Area"].values
        flux_temp = group["Flux"].values
        losv_temp = group["LOS_V"].values
        X_temp = group["X"].values
        Y_temp = group["Y"].values
        label_temp = group["label"].values
        frame_temp = group["frame"].values
        ecc_temp = group["ecc"].values

        if len(numpy.unique(label_temp)) > 1:
            raise ValueError("More than one label in the group")
        if numpy.any(numpy.diff(frame_temp) != 1):
            raise ValueError("Frames are not consecutive")

        area_tot.append(area_temp)
        flux_tot.append(flux_temp)
        losv_tot.append(losv_temp)
        X_tot.append(X_temp)
        Y_tot.append(Y_temp)
        label_tot.append(label_temp)
        frame_tot.append(frame_temp)
        ecc_tot.append(ecc_temp)

    df_final = pandas.DataFrame({
        "label": [x[0] for x in label_tot],
        "Lifetime": [len(x) for x in frame_tot],
        "X": X_tot, "Y": Y_tot,
        "Area": area_tot, "Flux": flux_tot,
        "LOS_V": losv_tot, "Frames": frame_tot, "ecc": ecc_tot
    })
    df_final = df_final[df_final["Lifetime"] >= minliftime]

    vxtot, vytot, stdvxtot, stdvytot = [], [], [], []
    for j in tqdm.tqdm(range(len(df_final)), desc="Computing velocities"):
        x = numpy.array(df_final["X"].iloc[j])
        y = numpy.array(df_final["Y"].iloc[j])
        vx = numpy.gradient(x) * dx / dt
        vy = numpy.gradient(y) * dx / dt
        vxtot.append(vx)
        vytot.append(vy)
        stdvxtot.append(numpy.std(vx))
        stdvytot.append(numpy.std(vy))

    df_final["Vx"] = vxtot
    df_final["Vy"] = vytot
    df_final["stdVx"] = stdvxtot
    df_final["stdVy"] = stdvytot
    df_final = df_final.reset_index(drop=True)
    return df_final


###################################
###### DEPRECATED FUNCTIONS #######
###################################


def tabulation(files: str, filesB: str, dx: float, dt: float, cores: int, minliftime: int = 4) -> pandas.DataFrame:
    """Deprecated: use tabulation_parallel instead."""

    df = pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "frame", "ecc"])
    img = astropy.io.fits.getdata(filesB[0], memmap=False)
    size = numpy.shape(img)
    x_1, y_1 = numpy.meshgrid(numpy.arange(size[1]), numpy.arange(size[0]))
    for j, file in tqdm.tqdm(enumerate(files), desc="Tabulation"):
        src_img = astropy.io.fits.getdata(filesB[j], memmap=False)
        asc_img = astropy.io.fits.getdata(file, memmap=False)
        unique_ids = numpy.unique(asc_img)
        for i in unique_ids:
            if i == 0:
                continue
            mask = (asc_img == i)
            Bm = src_img * mask
            Area = mask.sum()
            Flux = Bm.sum() / Area
            X = ((mask * x_1) * Bm).sum() / Bm.sum()
            Y = ((mask * y_1) * Bm).sum() / Bm.sum()
            r = numpy.sqrt(Area / numpy.pi)
            circle = ((x_1 - X) ** 2 + (y_1 - Y) ** 2 < r ** 2).astype(int) * mask
            ecc = circle.sum() / Area
            temp = pandas.DataFrame([[i, X, Y, Area, Flux, j, ecc]], columns=["label", "X", "Y", "Area", "Flux", "frame", "ecc"])
            df = pandas.concat([df, temp], ignore_index=False)

    groups = df.groupby("label")
    area_tot, flux_tot, X_tot, Y_tot, label_tot, frame_tot, ecc_tot = [], [], [], [], [], [], []

    for name, group in tqdm.tqdm(groups, desc="Merging common labels"):
        area_temp = group["Area"].values
        flux_temp = group["Flux"].values
        X_temp = group["X"].values
        Y_temp = group["Y"].values
        label_temp = group["label"].values
        frame_temp = group["frame"].values
        ecc_temp = group["ecc"].values

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
        ecc_tot.append(ecc_temp)

    df_final = pandas.DataFrame({
        "label": [x[0] for x in label_tot],
        "Lifetime": [len(x) for x in frame_tot],
        "X": X_tot, "Y": Y_tot,
        "Area": area_tot, "Flux": flux_tot,
        "Frames": frame_tot, "ecc": ecc_tot
    })
    df_final = df_final[df_final["Lifetime"] >= minliftime]

    vxtot, vytot, stdvxtot, stdvytot = [], [], [], []
    for j in tqdm.tqdm(range(len(df_final)), desc="Computing velocities"):
        x = numpy.array(df_final["X"].iloc[j])
        y = numpy.array(df_final["Y"].iloc[j])
        vx = numpy.gradient(x) * dx / dt
        vy = numpy.gradient(y) * dx / dt
        vxtot.append(vx)
        vytot.append(vy)
        stdvxtot.append(numpy.std(vx))
        stdvytot.append(numpy.std(vy))

    df_final["Vx"] = vxtot
    df_final["Vy"] = vytot
    df_final["stdVx"] = stdvxtot
    df_final["stdVy"] = stdvytot
    df_final = df_final.reset_index(drop=True)
    return df_final