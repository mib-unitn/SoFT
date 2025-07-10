import numpy
import astropy
import scipy
import skimage
import pandas
import os
import glob
from numba import jit
from tqdm import tqdm
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


def track_all_LF(datapath, cores, l_thr, sign, m_size, dx, dt, N, verbose=False):
    # create a large datasets pipeline for SoFT
    # Load the data
    data = sorted(glob.glob(datapath+"00-data/*.fits"))
    # Set the number of workers
    number_of_workers = numpy.min([len(data), cores])
    housekeeping_LF(datapath)

    with multiprocessing.Pool(number_of_workers) as p:
        p.starmap(process_image_ss, [(datapath, img, l_thr,sign, m_size, verbose) for img in data])
    # Assign unique IDs
    id_data = sorted(glob.glob(datapath+"02-id/*.fits"))
    unique_id(id_data, datapath, verbose)

    # Start the association
    img_n0 = numpy.array(astropy.io.fits.open(id_data[0])[0].data)
    astropy.io.fits.writeto(datapath+f"03-assoc/{id_data[0].split('/')[-1]}", img_n0, overwrite=True)
    # Now we can start matching features
    for j in range(len(id_data) - 1):
        img_n0 = numpy.array(astropy.io.fits.open(datapath+f"03-assoc/{id_data[j].split('/')[-1]}", ignore_missing_simple=True, ignore_missing_end=True, ignore_blank=True, output_verify="ignore")[0].data, dtype=numpy.int64)
        img_n1 = numpy.array(astropy.io.fits.open(id_data[j + 1], ignore_missing_simple=True, ignore_missing_end=True, ignore_blank=True, output_verify="ignore")[0].data, dtype=numpy.int64)
        matches_1, matches_2 = back_and_forth_matching_LF(img_n0, img_n1)
        img_assoc = associate_LF(img_n1, matches_1, matches_2)
        astropy.io.fits.writeto(datapath+f"03-assoc/{id_data[j + 1].split('/')[-1]}", img_assoc, overwrite=True)


    asc_files_total = sorted(glob.glob(os.path.join(datapath,"03-assoc/*.fits")))
    src_files_total = sorted(glob.glob(os.path.join(datapath+"00-data/*.fits")))
    doppler_files_total = sorted(glob.glob(os.path.join(datapath+"00b-doppler/*.fits")))

    # split the files in N different groups
    asc_files = numpy.array_split(asc_files_total, N)
    src_files = numpy.array_split(src_files_total, N)
    doppler_files = numpy.array_split(doppler_files_total, N)

    # do tabulation in parallel for each group, saving the results in a dataframe and then merging them, keep track of the N/total so that they can be latwer on ordered and merged
    for i in tqdm(range(N)):
        if os.path.exists(os.path.join(datapath+f"04-dataframes/dataframe_{i}.parquet")):
            continue
        else:
            df = tabulation_parallel_doppler(asc_files[i], doppler_files[i], src_files[i], dx, dt, cores)
            df.to_json(os.path.join(datapath+f"04-dataframes/dataframe_{i}.json"))


@jit(nopython=True)
def back_and_forth_matching_LF(file1, file2):
    unique_id_1 = numpy.unique(file1)
    unique_id_1 = unique_id_1[unique_id_1 != 0]
    unique_id_2 = numpy.unique(file2)
    unique_id_2 = unique_id_2[unique_id_2 != 0]
    # ccreate two empty 1D arrays to store the forward_1 and forward_2 matches
    forward_matches_1 = numpy.empty(0)
    forward_matches_2 = numpy.empty(0)
    for id_1 in unique_id_1:
        wh1 = numpy.where(file1 == id_1)
        set1 = set(zip(wh1[0], wh1[1]))
        max_intersection_size = 0
        for id_2 in unique_id_2:
            wh2 = numpy.where(file2 == id_2)
            set2 = set(zip(wh2[0], wh2[1]))
            temp_intersection_size = len(set1.intersection(set2))
            if temp_intersection_size > max_intersection_size:
                max_intersection_size = temp_intersection_size
                best_match_1 = id_1
                best_match_2 = id_2
        if max_intersection_size != 0:
            forward_matches_1 = numpy.append(forward_matches_1, best_match_1)
            forward_matches_2 = numpy.append(forward_matches_2, best_match_2)

    backward_matches_1 = numpy.empty(0)
    backward_matches_2 = numpy.empty(0)
    for id_2 in unique_id_2:
        wh2 = numpy.where(file2 == id_2)
        set2 = set(zip(wh2[0], wh2[1]))
        max_intersection_size = 0
        for id_1 in unique_id_1:
            wh1 = numpy.where(file1 == id_1)
            set1 = set(zip(wh1[0], wh1[1]))
            temp_intersection_size = len(set1.intersection(set2))
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
    return mutual_matches_1, mutual_matches_2

def associate_LF(img_n1, matches_1, matches_2):
    for idx in range(len(matches_1)):
        numpy.place(img_n1, img_n1 == matches_2[idx], matches_1[idx])
    return img_n1



def tabulation_parallel_doppler(files: str, filesD: str, filesB: str, dx: float, dt: float, cores: int, minliftime: int = 4) -> pandas.DataFrame:
    def process_file(j):
        file = files[j]
        src_img = astropy.io.fits.getdata(filesB[j], memmap=False)
        asc_img = astropy.io.fits.getdata(file, memmap=False)
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
            LosV = LosV_s #numpy.nanmean(LosV_s)
            X = ((mask * x_1) * Bm).sum() / Bm.sum()
            Y = ((mask * y_1) * Bm).sum() / Bm.sum()
            r = numpy.sqrt(Area / numpy.pi)
            circle = (x_1 - X)**2 + (y_1 - Y)**2 < r**2
            circle = circle.astype(int)
            circle = circle * mask
            Area_circle = circle.sum()
            ecc = Area_circle / Area
            temp = pandas.DataFrame([[i, X, Y, Area, Flux, LosV, j, ecc]], columns=["label", "X", "Y", "Area", "Flux", "LOS_V", "frame", "ecc"])
            df_temp = pandas.concat([df_temp, temp], ignore_index=False)
        return df_temp

    df = pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "LOS_V", "frame", "ecc"])
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
    losv_tot = []
    X_tot = []
    Y_tot = []
    label_tot = []
    frame_tot = []
    ecc_tot = []

    for name, group in groups:
        area_temp = group["Area"].values
        flux_temp = group["Flux"].values
        losv_temp = group["LOS_V"].values
        X_temp = group["X"].values
        Y_temp = group["Y"].values
        label_temp = group["label"].values
        frame_temp = group["frame"].values
        ecc_temp = group["ecc"].values

        # Perform some sanity checks
        if len(area_temp) != len(flux_temp):
            raise ValueError("area and flux have different lengths")
        if len(area_temp) != len(losv_temp):
            raise ValueError("area and losv have different lengths")
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
        losv_tot.append(losv_temp)
        X_tot.append(X_temp)
        Y_tot.append(Y_temp)
        label_tot.append(label_temp)
        frame_tot.append(frame_temp)
        ecc_tot.append(ecc_temp)

    df_final = pandas.DataFrame(columns=["label", "Lifetime", "X", "Y", "Area", "Flux", "LOS_V", "Frames", "ecc"])
    df_final["label"] = [x[0] for x in label_tot]
    df_final["Lifetime"] = [len(x) for x in frame_tot]
    df_final["X"] = X_tot
    df_final["Y"] = Y_tot
    df_final["Area"] = area_tot
    df_final["Flux"] = flux_tot
    df_final["LOS_V"] = losv_tot
    df_final["Frames"] = frame_tot
    df_final["ecc"] = ecc_tot
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


def img_pre_pos(img, l_thr):
    img_pos = img.copy()
    img_pos[img_pos < 0] = 0
    img_pos = numpy.array(img_pos, dtype=numpy.float64)
    img_pos[img_pos < l_thr] = 0
    return img_pos

def img_pre_neg(img, l_thr):
    img_neg = img.copy()
    img_neg = -1*numpy.array(img_neg, dtype=numpy.float64)
    img_neg[img_neg < 0] = 0
    img_neg[img_neg < l_thr] = 0
    return img_neg

def simple_labels(img):
    # simple labels rather than too exotic segmentation techniques as we don't want to risk the code splitting the sunspots' umbra
    labels = scipy.ndimage.label(img)
    return labels



def detection_ss(img, l_thr, sign="both"):
    
    img = numpy.array(img)
    if sign == "both":
        img_pos = img_pre_pos(img, l_thr)
        img_neg = img_pre_neg(img, l_thr)
        labels_pos,_ = simple_labels(img_pos)
        labels_neg,_ = simple_labels(img_neg)
        labels_neg = -1*labels_neg
        labels = labels_pos + labels_neg
    elif sign == "pos":
        img_pos = img_pre_pos(img, l_thr)
        labels,_ = simple_labels(img_pos)
    elif sign == "neg":
        img_neg = img_pre_neg(img, l_thr)
        labels,_ = simple_labels(img_neg)
        labels = -1*labels
    else:
        raise ValueError("sign must be either 'both', 'pos' or 'neg'")
    return labels

def process_image_ss(datapath: str, data: str, l_thr: int, sign, min_size:int=4, verbose=False) -> None:

    image = astropy.io.fits.getdata(data, memmap=False)
    labels = detection_ss(image, l_thr, sign)
    astropy.io.fits.writeto(datapath+f"01-mask/{data.split(os.sep)[-1]}", labels, overwrite=True)
    labels = identification(labels, min_size=4, verbose=verbose)
    astropy.io.fits.writeto(datapath+f"02-id/{data.split(os.sep)[-1]}", labels, overwrite=True)

def identification(labels, min_size, verbose=False):

    count = 0
    uid = numpy.unique(labels)
    original_number = len(uid)
    if verbose:
        print(f"Number of clumps detected: {original_number-1}")

    for k in tqdm(uid, leave=False):
        sz = numpy.where(labels == k)
        if len(sz[0]) < min_size:
            labels = numpy.where(labels == k, 0, labels)
            count+=1

    num = original_number - count
    if verbose:
        print(f"Number of clumps surviving the identification process: {num}")
    if num == 0:
        raise ValueError("No clumps survived the identification process")
    else:
        if verbose:
            print(f"Number of clumps surviving the identification process: {num}")
        pass

    return labels


def unique_id(id_data, datapath, verbose):

    u_id_p = 1
    u_id_n = -1
    for filename in tqdm(id_data):
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
    if verbose:
        print(f"Total number of unique IDs: {u_id_p+abs(u_id_n)-1}")


def housekeeping_LF(datapath: str) -> None:

    if not os.path.exists(datapath+"01-mask") and not os.path.exists(datapath+"02-id") and not os.path.exists(datapath+"03-assoc") and not os.path.exists(datapath+"04-dataframes"):
        os.makedirs(datapath+"01-mask")
        os.makedirs(datapath+"02-id")
        os.makedirs(datapath+"03-assoc")
        os.makedirs(datapath+"04-dataframes")
    else:
        files_mask = glob.glob(datapath+"01-mask/*")
        files_id = glob.glob(datapath+"02-id/*")
        files_assoc = glob.glob(datapath+"03-assoc/*")
        files_dataframes = glob.glob(datapath+"04-dataframes/*")
        if len(files_mask) == len(files_id) == len(files_assoc) != 0:
            response = "y"
            if response == "y":
                for file in files_mask:
                    os.remove(file)
                for file in files_id:
                    os.remove(file)
                for file in files_assoc:
                    os.remove(file)
                for file in files_dataframes:
                    os.remove(file)
            else:
                pass
        elif len(files_mask) != len(files_id):
            print("The number of files in the directories 01-mask and 02-id do not match. Deleting all files.")
            for file in files_mask:
                os.remove(file)
            for file in files_id:
                os.remove(file)
        else:
            pass
            print("The directories are empty.")