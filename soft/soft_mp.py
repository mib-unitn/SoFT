import numpy
import astropy.io.fits
import scipy
import skimage
import pandas
import os
import glob
import time
from typing import Union
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def track_all(datapath: str, cores: int, min_distance: int, l_thr: float, min_size: int, dx: float, dt: float, sign: str, separation: bool, verbose:bool=False, doppler:bool =False) -> None:
    data = sorted(glob.glob(datapath+"00-data/*.fits"))
    number_of_workers = numpy.min([len(data), cores])
    print(color.RED + color.BOLD + f"Number of cores used: {number_of_workers}" + color.END)
    start = time.time()
    print(color.RED + color.BOLD + "Cleaning up" + color.END)
    housekeeping(datapath)

    print(color.RED + color.BOLD + "Detecting features..." + color.END)
    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        futures = [executor.submit(process_image, datapath, img_file, l_thr, min_distance, sign, separation, min_size, verbose)
                   for img_file in data]
        for f in as_completed(futures):
            f.result()

    print(color.RED + color.BOLD + "Assigning unique IDs..." + color.END)
    id_data = sorted(glob.glob(datapath+"02-id/*.fits"))
    unique_id(id_data, datapath, verbose)
    print(color.GREEN + color.BOLD + "Feature detection step ended" + color.END)

    print(color.RED + color.BOLD + "Associating features..." + color.END)
    associate(datapath, verbose)

    print(color.RED + color.BOLD + "Cleaning up" + color.END)
    os.system(f"rm -rf {datapath}temp*")

    print(color.RED + color.BOLD + "Starting tabulation" + color.END)
    asc_files = sorted(glob.glob(os.path.join(datapath,"03-assoc/*.fits")))
    src_files = sorted(glob.glob(os.path.join(datapath+"00-data/*.fits")))
    if doppler:
        print(color.RED + color.BOLD + "Doppler images are not supported in the process-based version of SoFT!" + color.END)
    else:
        df = tabulation_parallel(asc_files, src_files, dx, dt, cores)

    df.to_json(os.path.join(datapath+"dataframe.json"))
    end = time.time()
    print(color.GREEN + color.BOLD + "Dataframe saved" + color.END)
    print(color.YELLOW + color.BOLD + f"Number of elements tracked: {len(df)}" + color.END)
    print(color.PURPLE + color.BOLD + f"Time elapsed: {end-start:.2f} seconds" + color.END)


###################################
##### HELPER FUNCTIONS ############
###################################

def housekeeping(datapath: str) -> None:
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
            for file in files_mask + files_id + files_assoc:
                os.remove(file)
            print(color.BOLD + color.GREEN + "Files deleted successfully." + color.END)
        elif len(files_mask) != len(files_id):
            print("The number of files in the directories 01-mask and 02-id do not match. Deleting all files.")
            for file in files_mask + files_id:
                os.remove(file)
        else:
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
    else:
        coords = skimage.feature.peak_local_max(img, min_distance=min_dist)
        mask = numpy.zeros(img.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = scipy.ndimage.label(mask)
        labels = skimage.segmentation.watershed(-img, markers, mask=img, compactness=0.001)
        labels = numpy.array(labels, dtype=numpy.float64)
        return labels, coords


###################################
##### MAIN FUNCTIONS ##############
###################################

def detection(img: numpy.ndarray, l_thr: float, min_distance:int,sign:str="both", separation:bool=False, verbose:bool=False) -> Union[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]:
    img = numpy.array(img)
    if sign == "both":
        img_pos = img_pre_pos(img, l_thr)
        img_neg = img_pre_neg(img, l_thr)
        labels_pos,_ = watershed_routine(img_pos, min_distance, separation)
        labels_neg,_ = watershed_routine(img_neg, min_distance, separation)
        labels_neg = -1*labels_neg
        labels = labels_pos + labels_neg
        if verbose:
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


def identification(labels: numpy.ndarray, min_size: int, verbose:bool = False) -> numpy.ndarray:
    count = 0
    uid = numpy.unique(labels)
    original_number = len(uid)
    if verbose:
        print(f"Number of clumps detected: {original_number-1}")

    for k in uid:
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


def process_image(datapath: str, data: str, l_thr: float, min_distance: int, sign:str="both", separation:bool=True, min_size:int=4, verbose:bool=False) -> None:
    image = astropy.io.fits.getdata(data, memmap=False)
    labels = detection(image, l_thr, min_distance, sign=sign, separation=separation, verbose=verbose)
    astropy.io.fits.writeto(datapath+f"01-mask/{data.split(os.sep)[-1]}", labels, overwrite=True)
    labels = identification(labels, min_size, verbose=verbose)
    astropy.io.fits.writeto(datapath+f"02-id/{data.split(os.sep)[-1]}", labels, overwrite=True)


def unique_id(id_data: str, datapath: str, verbose:bool=False) -> None:
    u_id_p = 1
    u_id_n = -1
    for filename in id_data:
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
        print(f"Total number of unique IDs: {u_id_p+abs(u_id_n)-1}")


def array_row_intersection(a: numpy.ndarray,b:numpy.ndarray) -> numpy.ndarray:
    tmp=numpy.prod(numpy.swapaxes(a[:,:,None],1,2)==b,axis=2)
    return a[numpy.sum(numpy.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]


def back_and_forth_matching_PARALLEL(fname1: str, fname2: str, round: int, datapath: str, verbose:bool=False) -> None:
    cube1 = astropy.io.fits.getdata(fname1, memmap=False)
    cube2 = astropy.io.fits.getdata(fname2, memmap=False)

    file1 = cube1[-1] if cube1.ndim > 2 else cube1
    file2 = cube2[0] if cube2.ndim > 2 else cube2

    unique_id_1 = numpy.unique(file1)
    unique_id_1 = unique_id_1[unique_id_1 != 0]
    forward_matches_1 = numpy.empty(0)
    forward_matches_2 = numpy.empty(0)
    for id_1 in unique_id_1:
        try:
            wh1 = numpy.where(file1 == id_1)
            set1 = numpy.stack((wh1[0], wh1[1])).T
        except:
            print(f"Error in forward matching for id_1: {id_1}. Skipping.")
            print(f"Frame was {fname1.split(os.sep)[-1]} and round was {round}")
            raise ValueError("Error in forward matching. Check the input files.")
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
    for id_2 in unique_id_2:
        try:
            wh2 = numpy.where(file2 == id_2)
            set2 = numpy.stack((wh2[0], wh2[1])).T
        except:
            print(f"Error in backward matching for id_2: {id_2}. Skipping.")
            print(f"Frame was {fname2.split(os.sep)[-1]} and round was {round}")
            raise ValueError("Error in backward matching. Check the input files.")
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
    for kk in range(len(forward_matches_1)):
        if forward_matches_1[kk] in backward_matches_1 and forward_matches_2[kk] in backward_matches_2:
            fwm1 = forward_matches_1[kk]
            fwm2 = forward_matches_2[kk]
            mutual_matches_1 = numpy.append(mutual_matches_1, fwm1)
            mutual_matches_2 = numpy.append(mutual_matches_2, fwm2)
    
    for idx in range(len(mutual_matches_1)):
        numpy.place(cube2, cube2 == mutual_matches_2[idx], mutual_matches_1[idx])
    
    if len(numpy.shape(cube1)) == 2:
        cube1 = cube1.reshape(1, cube1.shape[0], cube1.shape[1])
    if len(numpy.shape(cube2)) == 2:
        cube2 = cube2.reshape(1, cube2.shape[0], cube2.shape[1])

    cube2 = numpy.concatenate((cube1, cube2), axis=0)

    astropy.io.fits.writeto(datapath+f"temp{round}/{fname1.split(os.sep)[-1]}", cube2, overwrite=True)
    if verbose:
        print(color.YELLOW + f"Done with {fname1.split(os.sep)[-1]}, {fname2.split(os.sep)[-1]}" + color.END)


def associate(datapath: str, verbose:bool=False) -> None:
    number_of_workers = os.cpu_count()
    id_data = sorted(glob.glob(datapath+"02-id/*.fits"))
    round = 0
    os.makedirs(datapath+"temp0", exist_ok=True)
    subgroups = [id_data[i:i+2] for i in range(0, len(id_data), 2)]
    if len(subgroups[-1]) == 1:
        img = astropy.io.fits.getdata(subgroups[-1][0], memmap=False)
        img = img.reshape(1, img.shape[0], img.shape[1])
        astropy.io.fits.writeto(datapath+f"temp{round}/{subgroups[-1][0].split(os.sep)[-1]}", img, overwrite=True)
        subgroups = subgroups[:-1]

    print(color.RED + color.BOLD + "Starting the first round of association" + color.END)
    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        futures = [executor.submit(back_and_forth_matching_PARALLEL, subgroup[0], subgroup[1], round, datapath, verbose)
                   for subgroup in subgroups]
        for f in as_completed(futures):
            f.result()

    max_iter = 10
    for round in range(1, max_iter):
        data = sorted(glob.glob(datapath+f"temp{round-1}/*.fits"))
        os.makedirs(datapath+f"temp{round}", exist_ok=True)
        if len(data) < 2:
            break
        subgroups = [data[i:i+2] for i in range(0, len(data), 2)]
        if len(subgroups[-1]) == 1:
            img = astropy.io.fits.getdata(subgroups[-1][0], memmap=False)
            img = img.reshape(img.shape[0], img.shape[1], img.shape[2])
            astropy.io.fits.writeto(datapath+f"temp{round}/{subgroups[-1][0].split(os.sep)[-1]}", img, overwrite=True)
            subgroups = subgroups[:-1]

        print(color.RED + color.BOLD + f"Starting the {round+1} round of association" + color.END)
        with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
            futures = [executor.submit(back_and_forth_matching_PARALLEL, subgroup[0], subgroup[1], round, datapath, verbose)
                       for subgroup in subgroups]
            for f in as_completed(futures):
                f.result()

    data = astropy.io.fits.getdata(sorted(glob.glob(datapath+f"temp{round-1}/*.fits"))[0], memmap=False)
    for i in range(data.shape[0]):
        astropy.io.fits.writeto(datapath+f"03-assoc/{i:04d}.fits", data[i, :, :], overwrite=True)

    print(color.GREEN + color.BOLD + "Finished association" + color.END)

def _tabulation_worker(args):
    j, file, fileB, dx, dt = args
    src_img = astropy.io.fits.getdata(fileB, memmap=False)
    asc_img = astropy.io.fits.getdata(file, memmap=False)
    size = src_img.shape
    x_1, y_1 = numpy.meshgrid(numpy.arange(size[1]), numpy.arange(size[0]))

    results = []
    for i in numpy.unique(asc_img):
        if i == 0:
            continue
        mask = (asc_img == i)
        Area = mask.sum()
        if Area == 0:
            continue
        Bm = src_img * mask
        Bsum = Bm.sum()
        if Bsum == 0:
            continue
        X = ((mask * x_1) * Bm).sum() / Bsum
        Y = ((mask * y_1) * Bm).sum() / Bsum
        r = numpy.sqrt(Area / numpy.pi)
        circle = (((x_1 - X)**2 + (y_1 - Y)**2) < r**2).astype(int) * mask
        Area_circle = circle.sum()
        ecc = Area_circle / Area
        Flux = Bsum / Area
        results.append([i, X, Y, Area, Flux, j, ecc])

    if results:
        return pandas.DataFrame(results, columns=["label", "X", "Y", "Area", "Flux", "frame", "ecc"])
    else:
        return pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "frame", "ecc"])


def tabulation_parallel(files: list, filesB: list, dx: float, dt: float, cores: int, minliftime: int = 4) -> pandas.DataFrame:
    with ProcessPoolExecutor(max_workers=cores) as executor:
        tasks = [(j, files[j], filesB[j], dx, dt) for j in range(len(files))]
        results = list(executor.map(_tabulation_worker, tasks))

    df = pandas.concat([r for r in results if not r.empty], ignore_index=False)

    groups = df.groupby("label")
    area_tot, flux_tot, X_tot, Y_tot, label_tot, frame_tot, ecc_tot = [], [], [], [], [], [], []

    for name, group in groups:
        area_temp, flux_temp = group["Area"].values, group["Flux"].values
        X_temp, Y_temp = group["X"].values, group["Y"].values
        label_temp, frame_temp, ecc_temp = group["label"].values, group["frame"].values, group["ecc"].values

        if numpy.any(numpy.diff(frame_temp) != 1):
            raise ValueError("Frames are not consecutive")

        area_tot.append(area_temp)
        flux_tot.append(flux_temp)
        X_tot.append(X_temp)
        Y_tot.append(Y_temp)
        label_tot.append(label_temp)
        frame_tot.append(frame_temp)
        ecc_tot.append(ecc_temp)

    df_final = pandas.DataFrame(columns=["label", "Lifetime", "X", "Y", "Area", "Flux", "Frames", "ecc"])
    df_final["label"] = [x[0] for x in label_tot]
    df_final["Lifetime"] = [len(x) for x in frame_tot]
    df_final["X"] = X_tot
    df_final["Y"] = Y_tot
    df_final["Area"] = area_tot
    df_final["Flux"] = flux_tot
    df_final["Frames"] = frame_tot
    df_final["ecc"] = ecc_tot
    df_final = df_final[df_final["Lifetime"] >= minliftime]

    vxtot, vytot, stdvxtot, stdvytot = [], [], [], []
    for j in range(len(df_final)):
        x, y = numpy.array(df_final["X"].iloc[j]), numpy.array(df_final["Y"].iloc[j])
        vx = numpy.gradient(x) * dx / dt
        vy = numpy.gradient(y) * dx / dt
        stdx, stdy = numpy.std(vx), numpy.std(vy)
        vxtot.append(vx)
        vytot.append(vy)
        stdvxtot.append(stdx)
        stdvytot.append(stdy)

    df_final.loc[:, "Vx"] = vxtot
    df_final.loc[:, "Vy"] = vytot
    df_final.loc[:, "stdVx"] = stdvxtot
    df_final.loc[:, "stdVy"] = stdvytot


    df_final = df_final.reset_index(drop=True)
    return df_final
