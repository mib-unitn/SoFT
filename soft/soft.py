import numpy
import astropy.io.fits
import scipy
import scipy.spatial
import skimage
import pandas
import os
import glob
import re
import multiprocessing
import concurrent.futures
import time
from typing import Union
from pathos.multiprocessing import ProcessingPool
from fastprogress import master_bar, progress_bar, force_console_behavior

def _in_notebook() -> bool:
    """Return True when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

if not _in_notebook():
    force_console_behavior()   # clean single-line bars in plain terminals


class _ColorFormatter:
    """
    Context-aware color/style formatter.
    Returns ANSI escape codes in a terminal, HTML tags in a Jupyter notebook.

    Usage is identical to the old ``color`` class::

        color.RED + color.BOLD + "text" + color.END
    """

    _STYLES: dict[str, tuple[str, str]] = {
        # name       ANSI code          HTML open tag
        "PURPLE":    ("\033[95m", '<span style="color:#8e44ad">'),
        "CYAN":      ("\033[96m", '<span style="color:#1abc9c">'),
        "DARKCYAN":  ("\033[36m", '<span style="color:#16a085">'),
        "BLUE":      ("\033[94m", '<span style="color:#2980b9">'),
        "GREEN":     ("\033[92m", '<span style="color:#27ae60">'),
        "YELLOW":    ("\033[93m", '<span style="color:#f39c12">'),
        "RED":       ("\033[91m", '<span style="color:#e74c3c">'),
        "BOLD":      ("\033[1m",  "<b>"),
        "UNDERLINE": ("\033[4m",  "<u>"),
        # END closes every open tag; extra </b></span> is harmless in HTML
        "END":       ("\033[0m",  "</b></span>"),
    }

    def __getattr__(self, name: str) -> str:
        try:
            ansi, html = self._STYLES[name]
        except KeyError:
            raise AttributeError(f"_ColorFormatter has no attribute {name!r}")
        return html if _in_notebook() else ansi


color = _ColorFormatter()


def _mb_write(mb, text: str) -> None:
    """Write formatted text alongside a master_bar.

    fastprogress's NBMasterBar.write() outputs plain text (no HTML rendering),
    so in a notebook we call display(HTML(...)) directly — the same path used
    by _cprint — to get proper color/bold rendering.
    """
    if _in_notebook():
        from IPython.display import display, HTML
        display(HTML(text))
    else:
        mb.write(text)


def _cprint(*args, **kwargs) -> None:
    """Notebook-aware drop-in for ``print()`` with color-formatted strings.

    In a Jupyter notebook, renders the text as HTML so that the HTML tags
    produced by ``_ColorFormatter`` display correctly.  In a terminal,
    falls back to plain ``print()``.
    """
    if _in_notebook():
        from IPython.display import display, HTML
        text = " ".join(str(a) for a in args)
        display(HTML(f'<span style="font-family:monospace">{text}</span>'))
    else:
        print(*args, **kwargs)



def track_all(datapath: str, cores: int, l_thr: float, min_size: int, dx: float, dt: float, sign: str, separation: bool, verbose: bool = False, doppler: bool = False) -> None:

    """
    Executes a pipeline for feature detection, identification, association, tabulation, and data storage based on astronomical FITS files.

    Parameters:
    - datapath (str): Path to the main data directory.
    - cores (int): Number of CPU cores to utilize for parallel processing.
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

    start = time.time()

    # Master bar tracks the 5 top-level pipeline stages
    stages = ["Cleanup", "Detection", "Unique IDs", "Association", "Tabulation"]
    mb = master_bar(stages)
    _mb_write(mb, color.RED + color.BOLD + f"SoFT — {number_of_workers} core(s)" + color.END)

    for stage in mb:
        if stage == "Cleanup":
            _mb_write(mb, color.BOLD + "► Cleaning up output directories" + color.END)
            housekeeping(datapath)

        elif stage == "Detection":
            _mb_write(mb, color.BOLD + "► Detecting features..." + color.END)
            with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_workers) as pool:
                futures = [
                    pool.submit(process_image, datapath, img, l_thr, sign, separation, min_size, verbose)
                    for img in data
                ]
                for f in progress_bar(concurrent.futures.as_completed(futures),
                                      total=len(futures), parent=mb):
                    f.result()  # re-raise any worker exception

        elif stage == "Unique IDs":
            _mb_write(mb, color.BOLD + "► Assigning unique IDs..." + color.END)
            id_data = sorted(glob.glob(datapath + "02-id/*.fits"))
            unique_id(id_data, datapath, verbose, mb)
            _mb_write(mb, color.GREEN + "  Feature detection step ended" + color.END)

        elif stage == "Association":
            _mb_write(mb, color.BOLD + "► Associating features..." + color.END)
            associate(datapath, verbose, number_of_workers)
            os.system(f"rm -rf {datapath}temp*")

        elif stage == "Tabulation":
            _mb_write(mb, color.BOLD + "► Tabulating..." + color.END)
            asc_files = sorted(glob.glob(os.path.join(datapath, "03-assoc/*.fits")))
            src_files = sorted(glob.glob(os.path.join(datapath + "00-data/*.fits")))
            if len(asc_files) == 0 or len(src_files) == 0:
                _mb_write(mb, color.RED + "No association or source files found." + color.END)
                return
            if doppler:
                doppler_files = sorted(glob.glob(os.path.join(datapath + "00b-doppler/*.fits")))
                if len(doppler_files) == 0:
                    raise FileNotFoundError("No Doppler files found")
                df = tabulation_parallel_doppler(asc_files, doppler_files, src_files, dx, dt, cores, mb=mb)
            else:
                df = tabulation_parallel(asc_files, src_files, dx, dt, cores, mb=mb)

            df.to_json(os.path.join(datapath + "dataframe.json"))
            end = time.time()
            _mb_write(mb, color.GREEN + color.BOLD + "Dataframe saved" + color.END)
            _mb_write(mb, color.YELLOW + color.BOLD + f"Tracked features: {len(df)}" + color.END)
            _mb_write(mb, color.PURPLE + color.BOLD + f"Total time: {end - start:.1f} s" + color.END)

###################################
##### HELPER FUNCTIONS ############
###################################

def housekeeping(datapath: str) -> None:
    """
    Ensures the existence and proper state of specific directories and their
    contents within a given data path.
    """
    dirs = ["01-mask", "02-id", "03-assoc"]
    for d in dirs:
        os.makedirs(datapath + d, exist_ok=True)

    all_files = []
    for d in dirs:
        all_files.extend(glob.glob(datapath + d + "/*"))

    if all_files:
        _cprint(color.BOLD + color.RED + "WARNING: Output directories are not empty. Deleting files..." + color.END)
        for f in all_files:
            os.remove(f)
        _cprint(color.BOLD + color.GREEN + "Files deleted successfully." + color.END)
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


def _central_diff_velocity(pos: numpy.ndarray, dx: float, dt: float) -> numpy.ndarray:
    """Central finite differences with edge clamping.

    Interior points use ``(pos[i+1] - pos[i-1]) / 2``.  The first and last
    values are clamped to their nearest interior neighbour, avoiding the
    one-sided difference spike produced by ``numpy.gradient``.

    Returns an array of the same length as *pos*.
    """
    pos = numpy.asarray(pos, dtype=numpy.float64)
    v = numpy.empty_like(pos)
    if len(pos) >= 3:
        v[1:-1] = (pos[2:] - pos[:-2]) / 2.0 * dx / dt
        v[0] = v[1]
        v[-1] = v[-2]
    elif len(pos) == 2:
        v[:] = (pos[1] - pos[0]) * dx / dt
    else:
        v[:] = 0.0
    return v


def watershed_routine(img: numpy.ndarray, l_thr: float, sign: str, separation: bool = False) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Segments an image using a watershed algorithm.

    When separation=False, exactly one seed is placed per connected component
    at the distance-transform maximum — guaranteeing every blob above l_thr
    gets a label regardless of its peak intensity.

    When separation=True, multiple seeds per component are allowed, spaced
    adaptively based on the component's own distance transform, so that
    touching features can be split without losing weaker blobs.

    Parameters
    ----------
    img : ndarray
        Input 2D image.
    l_thr : float
        Lower intensity threshold; pixels below this are treated as background.
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
        # One seed per component at its EDT maximum.
        # scipy.ndimage.maximum_position does this in a single C-level pass
        # over the image — no Python loop over component pixels needed.
        comp_ids = sorted(all_components)
        seed_positions = scipy.ndimage.maximum_position(
            distance, labels=low_labels, index=comp_ids
        )
        # maximum_position returns a list of tuples when index is a list
        if isinstance(seed_positions, tuple):
            seed_positions = [seed_positions]
        coords = numpy.array(seed_positions, dtype=float)

    else:
        # Multiple seeds per component, greedily spaced by an adaptive
        # threshold derived from each component's own distance transform.
        # Uses a KD-tree for O(N log N) nearest-neighbour exclusion instead
        # of the previous O(N²) pure-Python distance loop.
        coords = []
        for comp in all_components:
            comp_mask = low_labels == comp
            comp_points = numpy.argwhere(comp_mask)          # (M, 2)
            vals = distance[comp_points[:, 0], comp_points[:, 1]]
            order = numpy.argsort(-vals)                     # descending EDT
            comp_points = comp_points[order]
            vals = vals[order]

            adaptive_spacing = max(3.0, 1.2 * float(vals[0]))

            seeds = [comp_points[0].astype(float)]           # always place peak
            for pt, v in zip(comp_points[1:], vals[1:]):
                # Build a KD-tree from accepted seeds and query once
                tree = scipy.spatial.KDTree(numpy.array(seeds))
                dist, _ = tree.query(pt.astype(float))
                if dist > adaptive_spacing:
                    seeds.append(pt.astype(float))

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


def detection(img: numpy.ndarray, l_thr: float, sign: str = "both", separation: bool = False, verbose: bool = False) -> numpy.ndarray:
    """
    Detects features in an image using a threshold and watershed algorithm.

    Parameters
    ----------
    img : ndarray
        Input image.
    l_thr : float
        Intensity threshold for detection. Pixels below this are ignored.
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
        labels_pos, _ = watershed_routine(img, l_thr, "pos", separation)
        labels_neg, _ = watershed_routine(img, l_thr, "neg", separation)
        labels_neg = -1 * labels_neg
        labels = labels_pos + labels_neg
        if verbose:
            print(f"Number of clumps detected: {len(numpy.unique(labels)) - 1}")
        return labels
    elif sign == "pos":
        labels_pos, _ = watershed_routine(img, l_thr, "pos", separation)
        return labels_pos
    elif sign == "neg":
        labels_neg, _ = watershed_routine(img, l_thr, "neg", separation)
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
    uids, counts = numpy.unique(labels, return_counts=True)
    original_number = len(uids)
    if verbose:
        print(f"Number of clumps detected: {original_number - 1}")

    # Single vectorised pass: zero out all labels whose pixel count < min_size.
    # Background (label == 0) always has count >= 1 so it is never zeroed.
    small = uids[(counts < min_size) & (uids != 0)]
    if small.size > 0:
        # Build a boolean mask of pixels that belong to small clumps
        kill_mask = numpy.isin(labels, small)
        labels = numpy.where(kill_mask, 0, labels)

    num_surviving = original_number - len(small)
    if num_surviving <= 1:   # only background survives
        raise ValueError("No clumps survived the identification process")
    if verbose:
        print(f"Number of clumps surviving the identification process: {num_surviving - 1}")

    return labels


def process_image(datapath: str, data: str, l_thr: float, sign: str = "both", separation: bool = True, min_size: int = 4, verbose: bool = False) -> None:
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
    labels = detection(image, l_thr, sign=sign, separation=separation, verbose=verbose)
    astropy.io.fits.writeto(datapath + f"01-mask/{data.split(os.sep)[-1]}", labels, overwrite=True)
    labels = identification(labels, min_size, verbose=verbose)
    astropy.io.fits.writeto(datapath + f"02-id/{data.split(os.sep)[-1]}", labels, overwrite=True)


def unique_id(id_data: str, datapath: str, verbose: bool = False, mb=None) -> None:
    """
    Assigns globally unique IDs to clumps across all frames.

    Performance: uses a single vectorised lookup-table remap (O(pixels + N))
    instead of N per-label full-image scans (O(N × pixels)).
    """
    u_id_p = 1
    u_id_n = -1
    for filename in progress_bar(id_data, parent=mb):
        img_n0 = astropy.io.fits.getdata(filename, memmap=False)
        ids = numpy.unique(img_n0[img_n0 != 0])
        ids_p = ids[ids > 0]
        ids_n = ids[ids < 0]

        # Build global remap counters first (no in-place risk)
        new_ids_p = numpy.arange(u_id_p, u_id_p + len(ids_p), dtype=img_n0.dtype)
        new_ids_n = numpy.arange(u_id_n, u_id_n - len(ids_n), -1, dtype=img_n0.dtype)
        u_id_p += len(ids_p)
        u_id_n -= len(ids_n)

        # Build a flat lookup table indexed by the original pixel value.
        # We shift all values by |min_id| so indices are non-negative.
        min_id = int(img_n0.min())          # may be 0 or negative
        offset = -min_id                    # shift to make all values >= 0
        lut_size = int(img_n0.max()) - min_id + 1
        lut = numpy.zeros(lut_size, dtype=img_n0.dtype)

        # Background (0) maps to 0; positive and negative ids get new values
        for old, new in zip(ids_p.astype(int), new_ids_p):
            lut[old + offset] = new
        for old, new in zip(ids_n.astype(int), new_ids_n):
            lut[old + offset] = new

        # Single vectorised remap: one pass over the whole image
        img_out = lut[img_n0.astype(int) + offset]

        astropy.io.fits.writeto(os.path.join(datapath, "02-id", os.path.basename(filename)), img_out, overwrite=True)
    if verbose:
        print(f"Total number of unique IDs: {u_id_p + abs(u_id_n) - 1}")


def array_row_intersection(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    """
    Returns rows of `a` that also appear in `b`.
    Uses set-based lookup for O(|a| + |b|) performance.
    """
    b_set = set(map(tuple, b))
    mask = numpy.array([tuple(row) in b_set for row in a])
    return a[mask]


def _run_matching(args: tuple) -> None:
    """Module-level wrapper so ProcessPoolExecutor can pickle the callable."""
    back_and_forth_matching_PARALLEL(*args)


def back_and_forth_matching_PARALLEL(fname1: str, fname2: str, round: int, datapath: str, verbose: bool = False) -> None:
    """
    Forward/backward matching of unique IDs between two FITS files,
    keeping only mutually consistent matches.

    Pixel-overlap counts between all (id1, id2) pairs are computed in one
    vectorised pass via numpy.unique on the stacked label pair array —
    O(pixels log pixels) instead of O(N1 × N2 × pixels).
    """
    cube1 = astropy.io.fits.getdata(fname1, memmap=False)
    cube2 = astropy.io.fits.getdata(fname2, memmap=False)

    file1 = cube1[-1] if cube1.ndim > 2 else cube1
    file2 = cube2[0]  if cube2.ndim > 2 else cube2

    ids1 = numpy.unique(file1); ids1 = ids1[ids1 != 0]
    ids2 = numpy.unique(file2); ids2 = ids2[ids2 != 0]

    if ids1.size == 0 or ids2.size == 0:
        # Nothing to match — just concatenate and write
        if cube1.ndim == 2:
            cube1 = cube1[numpy.newaxis, :, :]
        if cube2.ndim == 2:
            cube2 = cube2[numpy.newaxis, :, :]
        cube2 = numpy.concatenate((cube1, cube2), axis=0)
        astropy.io.fits.writeto(datapath + f"temp{round}/{fname1.split(os.sep)[-1]}", cube2, overwrite=True)
        return

    # --- Build co-occurrence table in one pass ---
    # Stack flattened label images: shape (2, H*W)
    flat1 = file1.ravel().astype(numpy.int64)
    flat2 = file2.ravel().astype(numpy.int64)

    # Keep only pixels where BOTH frames have a non-zero label
    both = (flat1 != 0) & (flat2 != 0)
    f1 = flat1[both]
    f2 = flat2[both]

    # Encode each (id1, id2) pair as a single int64 to count with unique
    # Use an offset so negative IDs work: shift to non-negative first
    min1 = int(f1.min()) if f1.size > 0 else 0
    min2 = int(f2.min()) if f2.size > 0 else 0
    off1 = -min1; off2 = -min2
    range2 = int(f2.max()) - min2 + 1 if f2.size > 0 else 1
    encoded = (f1.astype(numpy.int64) + off1) * range2 + (f2.astype(numpy.int64) + off2)
    pairs, counts = numpy.unique(encoded, return_counts=True)
    p1 = (pairs // range2).astype(numpy.int64) - off1
    p2 = (pairs %  range2).astype(numpy.int64) - off2

    # --- Forward best: for each id1, find id2 with max overlap ---
    forward_matches_1, forward_matches_2 = [], []
    for id1 in ids1:
        mask = (p1 == id1)
        if not mask.any():
            continue
        best_idx = numpy.argmax(counts[mask])
        forward_matches_2.append(int(p2[mask][best_idx]))
        forward_matches_1.append(int(id1))

    # --- Backward best: for each id2, find id1 with max overlap ---
    backward_matches_1, backward_matches_2 = [], []
    for id2 in ids2:
        mask = (p2 == id2)
        if not mask.any():
            continue
        best_idx = numpy.argmax(counts[mask])
        backward_matches_1.append(int(p1[mask][best_idx]))
        backward_matches_2.append(int(id2))

    # --- Mutual matching ---
    forward_pairs  = set(zip(forward_matches_1,  forward_matches_2))
    backward_pairs = set(zip(backward_matches_1, backward_matches_2))
    mutual_pairs   = forward_pairs & backward_pairs

    # --- Apply remap to cube2 via LUT (same pattern as unique_id) ---
    if mutual_pairs:
        mutual_matches_1 = numpy.array([p[0] for p in mutual_pairs], dtype=numpy.int64)
        mutual_matches_2 = numpy.array([p[1] for p in mutual_pairs], dtype=numpy.int64)

        min_c2  = int(cube2.min())
        off_c2  = -min_c2
        lut_sz  = int(cube2.max()) - min_c2 + 1
        lut     = numpy.arange(min_c2, min_c2 + lut_sz, dtype=numpy.int64)  # identity by default
        for old, new in zip(mutual_matches_2, mutual_matches_1):
            lut[int(old) + off_c2] = new
        cube2 = lut[cube2.astype(numpy.int64) + off_c2].astype(cube2.dtype)

    if cube1.ndim == 2:
        cube1 = cube1[numpy.newaxis, :, :]
    if cube2.ndim == 2:
        cube2 = cube2[numpy.newaxis, :, :]

    cube2 = numpy.concatenate((cube1, cube2), axis=0)
    astropy.io.fits.writeto(datapath + f"temp{round}/{fname1.split(os.sep)[-1]}", cube2, overwrite=True)
    if verbose:
        _cprint(color.YELLOW + f"Done with {fname1.split(os.sep)[-1]}, {fname2.split(os.sep)[-1]}" + color.END)


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
    _cprint(color.RED + color.BOLD + "Starting the first round of association" + color.END)
    args = [(subgroup[0], subgroup[1], round, datapath, verbose) for subgroup in subgroups]
    with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_workers) as pool:
        list(pool.map(_run_matching, args))

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
        _cprint(color.RED + color.BOLD + f"Starting the {round + 1} round of association" + color.END)
        args = [(subgroup[0], subgroup[1], round, datapath, verbose) for subgroup in subgroups]
        with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_workers) as pool:
            list(pool.map(_run_matching, args))


    final_files = sorted(glob.glob(datapath + f"temp{round - 1}/*.fits"))
    if not final_files:
        _cprint(color.RED + color.BOLD + "No associated files found." + color.END)
        return

    # Collect all cubes from the final temp directory and write them
    # frame-by-frame into 03-assoc/ with a global, monotonically
    # increasing frame index.  The original code only read final_files[0],
    # silently dropping every subsequent cube when the bisection did not
    # reduce to a single file.
    frame_idx = 0
    for fpath in final_files:
        data = astropy.io.fits.getdata(fpath, memmap=False)
        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]
        for i in range(data.shape[0]):
            astropy.io.fits.writeto(
                datapath + f"03-assoc/{frame_idx:04d}.fits",
                data[i, :, :],
                overwrite=True,
            )
            frame_idx += 1
    _cprint(color.GREEN + color.BOLD + "Finished association" + color.END)


def _tabulate_frame(args: tuple) -> pandas.DataFrame:
    """
    Module-level worker for tabulation_parallel — picklable by ProcessPoolExecutor.

    Receives a tuple (j, asc_path, src_path) so no closure variables are captured.
    x_1/y_1 grids are reconstructed from the image shape inside the worker.
    """
    j, asc_path, src_path = args
    src_img = astropy.io.fits.getdata(src_path, memmap=False).astype(numpy.float64)
    asc_img = astropy.io.fits.getdata(asc_path, memmap=False)

    ny, nx = src_img.shape
    x_1, y_1 = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))

    labels_present = numpy.unique(asc_img)
    labels_present = labels_present[labels_present != 0]
    if labels_present.size == 0:
        return pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "frame", "ecc"])

    index = list(labels_present)

    # Use unsigned magnetic signal as the centroid weight so that positive and
    # negative-polarity features are treated identically.  Keep the signed
    # signal separately for the reported mean flux density.
    src_clean = numpy.nan_to_num(src_img, nan=0.0, posinf=0.0, neginf=0.0)
    centroid_weights = numpy.abs(src_clean)

    area_vals = numpy.asarray(
        scipy.ndimage.sum(numpy.ones_like(src_clean), asc_img, index=index),
        dtype=numpy.float64,
    )
    signed_flux_sum = numpy.asarray(
        scipy.ndimage.sum(src_clean, asc_img, index=index),
        dtype=numpy.float64,
    )
    weight_sum = numpy.asarray(
        scipy.ndimage.sum(centroid_weights, asc_img, index=index),
        dtype=numpy.float64,
    )
    cx_vals = numpy.asarray(
        scipy.ndimage.sum(centroid_weights * x_1, asc_img, index=index),
        dtype=numpy.float64,
    )
    cy_vals = numpy.asarray(
        scipy.ndimage.sum(centroid_weights * y_1, asc_img, index=index),
        dtype=numpy.float64,
    )

    valid = (
        (area_vals > 0)
        & (weight_sum > 0)
        & numpy.isfinite(area_vals)
        & numpy.isfinite(signed_flux_sum)
        & numpy.isfinite(weight_sum)
        & numpy.isfinite(cx_vals)
        & numpy.isfinite(cy_vals)
    )
    if not numpy.any(valid):
        return pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "frame", "ecc"])

    area_vals = area_vals[valid]
    signed_flux_sum = signed_flux_sum[valid]
    weight_sum = weight_sum[valid]
    X_vals = cx_vals[valid] / weight_sum
    Y_vals = cy_vals[valid] / weight_sum
    Flux_vals = signed_flux_sum / area_vals
    lbl_valid  = numpy.array(index)[valid]
    r_vals     = numpy.sqrt(area_vals / numpy.pi)

    min_lbl  = int(asc_img.min())
    offset   = -min_lbl
    lut_size = int(asc_img.max()) - min_lbl + 1
    X_lut = numpy.zeros(lut_size, dtype=numpy.float64)
    Y_lut = numpy.zeros(lut_size, dtype=numpy.float64)
    r_lut = numpy.zeros(lut_size, dtype=numpy.float64)
    for lbl, X, Y, r in zip(lbl_valid, X_vals, Y_vals, r_vals):
        idx = int(lbl) + offset
        X_lut[idx] = X
        Y_lut[idx] = Y
        r_lut[idx] = r

    img_idx    = asc_img.astype(numpy.int64) + offset
    circle_all = ((x_1 - X_lut[img_idx])**2 + (y_1 - Y_lut[img_idx])**2 < r_lut[img_idx]**2).astype(numpy.float64)
    ecc_sums   = numpy.array(scipy.ndimage.sum(circle_all, asc_img, index=list(lbl_valid)))
    ecc_vals   = ecc_sums / area_vals

    records = [
        {"label": int(lbl), "X": X, "Y": Y, "Area": int(area),
         "Flux": flux, "frame": j, "ecc": ecc}
        for lbl, X, Y, area, flux, ecc in zip(
            lbl_valid, X_vals, Y_vals, area_vals, Flux_vals, ecc_vals)
    ]
    return pandas.DataFrame.from_records(records)


def _tabulate_frame_doppler(args: tuple) -> pandas.DataFrame:
    """
    Module-level worker for tabulation_parallel_doppler — picklable by ProcessPoolExecutor.
    """
    j, asc_path, src_path, dop_path = args
    src_img = astropy.io.fits.getdata(src_path, memmap=False).astype(numpy.float64)
    asc_img = astropy.io.fits.getdata(asc_path, memmap=False)
    alt_img = astropy.io.fits.getdata(dop_path, memmap=False).astype(numpy.float64)

    ny, nx = src_img.shape
    x_1, y_1 = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))

    labels_present = numpy.unique(asc_img)
    labels_present = labels_present[labels_present != 0]
    if labels_present.size == 0:
        return pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "LOS_V", "frame", "ecc"])

    index = list(labels_present)

    # Use unsigned magnetic signal as the centroid weight so that positive and
    # negative-polarity features are treated identically.  Keep the signed
    # signal separately for the reported mean flux density.
    src_clean = numpy.nan_to_num(src_img, nan=0.0, posinf=0.0, neginf=0.0)
    centroid_weights = numpy.abs(src_clean)

    area_vals = numpy.asarray(
        scipy.ndimage.sum(numpy.ones_like(src_clean), asc_img, index=index),
        dtype=numpy.float64,
    )
    signed_flux_sum = numpy.asarray(
        scipy.ndimage.sum(src_clean, asc_img, index=index),
        dtype=numpy.float64,
    )
    weight_sum = numpy.asarray(
        scipy.ndimage.sum(centroid_weights, asc_img, index=index),
        dtype=numpy.float64,
    )
    cx_vals = numpy.asarray(
        scipy.ndimage.sum(centroid_weights * x_1, asc_img, index=index),
        dtype=numpy.float64,
    )
    cy_vals = numpy.asarray(
        scipy.ndimage.sum(centroid_weights * y_1, asc_img, index=index),
        dtype=numpy.float64,
    )

    valid = (
        (area_vals > 0)
        & (weight_sum > 0)
        & numpy.isfinite(area_vals)
        & numpy.isfinite(signed_flux_sum)
        & numpy.isfinite(weight_sum)
        & numpy.isfinite(cx_vals)
        & numpy.isfinite(cy_vals)
    )
    if not numpy.any(valid):
        return pandas.DataFrame(columns=["label", "X", "Y", "Area", "Flux", "LOS_V", "frame", "ecc"])

    area_vals = area_vals[valid]
    signed_flux_sum = signed_flux_sum[valid]
    weight_sum = weight_sum[valid]
    X_vals = cx_vals[valid] / weight_sum
    Y_vals = cy_vals[valid] / weight_sum
    Flux_vals = signed_flux_sum / area_vals
    lbl_valid  = numpy.array(index)[valid]
    r_vals     = numpy.sqrt(area_vals / numpy.pi)

    min_lbl  = int(asc_img.min())
    offset   = -min_lbl
    lut_size = int(asc_img.max()) - min_lbl + 1
    X_lut = numpy.zeros(lut_size, dtype=numpy.float64)
    Y_lut = numpy.zeros(lut_size, dtype=numpy.float64)
    r_lut = numpy.zeros(lut_size, dtype=numpy.float64)
    for lbl, X, Y, r in zip(lbl_valid, X_vals, Y_vals, r_vals):
        idx = int(lbl) + offset
        X_lut[idx] = X
        Y_lut[idx] = Y
        r_lut[idx] = r

    img_idx    = asc_img.astype(numpy.int64) + offset
    circle_all = ((x_1 - X_lut[img_idx])**2 + (y_1 - Y_lut[img_idx])**2 < r_lut[img_idx]**2).astype(numpy.float64)
    ecc_sums   = numpy.array(scipy.ndimage.sum(circle_all, asc_img, index=list(lbl_valid)))
    ecc_vals   = ecc_sums / area_vals

    losv_list = [
        (lambda v: (v.__setitem__(v == 0, numpy.nan), v)[1])(alt_img[asc_img == lbl].copy())
        for lbl in lbl_valid
    ]

    records = [
        {"label": int(lbl), "X": X, "Y": Y, "Area": int(area),
         "Flux": flux, "LOS_V": losv, "frame": j, "ecc": ecc}
        for lbl, X, Y, area, flux, losv, ecc in zip(
            lbl_valid, X_vals, Y_vals, area_vals, Flux_vals, losv_list, ecc_vals)
    ]
    return pandas.DataFrame.from_records(records)


def tabulation_parallel(files: list, filesB: list, dx: float, dt: float, cores: int, minliftime: int = 4, mb=None) -> pandas.DataFrame:
    """
    Process segmentation maps and source FITS files in parallel.
    Extracts blob properties, tracks across frames, and computes velocities.

    Uses scipy.ndimage labelled statistics for O(pixels) per frame instead of
    O(N_features × pixels).
    """
    args = [(j, files[j], filesB[j]) for j in range(len(files))]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as pool:
        results = list(progress_bar(
            pool.map(_tabulate_frame, args),
            total=len(files), parent=mb, comment="Tabulation"
        ))

    df = pandas.concat(results, ignore_index=True)

    # Sort once globally, then groupby is already ordered — no per-group argsort needed
    df = df.sort_values(["label", "frame"]).reset_index(drop=True)

    groups = df.groupby("label", sort=False)
    area_tot, flux_tot, X_tot, Y_tot = [], [], [], []
    label_tot, frame_tot, ecc_tot = [], [], []

    for name, group in progress_bar(groups, total=groups.ngroups, parent=mb, comment="Merging"):
        if len(numpy.unique(group["label"].values)) > 1:
            raise ValueError(f"More than one label in group {name}")
        area_tot.append(group["Area"].values)
        flux_tot.append(group["Flux"].values)
        X_tot.append(group["X"].values)
        Y_tot.append(group["Y"].values)
        label_tot.append(group["label"].values)
        frame_tot.append(group["frame"].values)
        ecc_tot.append(group["ecc"].values)

    df_final = pandas.DataFrame({
        "label":    [x[0]  for x in label_tot],
        "Lifetime": [len(x) for x in frame_tot],
        "X": X_tot, "Y": Y_tot,
        "Area": area_tot, "Flux": flux_tot,
        "Frames": frame_tot, "ecc": ecc_tot
    })
    df_final = df_final[df_final["Lifetime"] >= minliftime]

    vxtot, vytot, stdvxtot, stdvytot = [], [], [], []
    for j in progress_bar(range(len(df_final)), parent=mb, comment="Velocities"):
        x  = numpy.array(df_final["X"].iloc[j])
        y  = numpy.array(df_final["Y"].iloc[j])
        vx = _central_diff_velocity(x, dx, dt)
        vy = _central_diff_velocity(y, dx, dt)
        vxtot.append(vx)
        vytot.append(vy)
        stdvxtot.append(numpy.std(vx))
        stdvytot.append(numpy.std(vy))

    df_final["Vx"]    = vxtot
    df_final["Vy"]    = vytot
    df_final["stdVx"] = stdvxtot
    df_final["stdVy"] = stdvytot
    df_final = df_final.reset_index(drop=True)
    return df_final


def tabulation_parallel_doppler(files: str, filesD: str, filesB: str, dx: float, dt: float, cores: int, minliftime: int = 4, mb=None) -> pandas.DataFrame:
    """
    Like tabulation_parallel but also extracts LOS (Doppler) velocity per feature.
    Uses scipy.ndimage labelled statistics for O(pixels) per frame.
    """
    args = [(j, files[j], filesB[j], filesD[j]) for j in range(len(files))]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as pool:
        results = list(progress_bar(
            pool.map(_tabulate_frame_doppler, args),
            total=len(files), parent=mb, comment="Tabulation (Doppler)"
        ))

    df = pandas.concat(results, ignore_index=True)
    df = df.sort_values(["label", "frame"]).reset_index(drop=True)

    groups = df.groupby("label", sort=False)
    area_tot, flux_tot, losv_tot, X_tot, Y_tot, label_tot, frame_tot, ecc_tot = [], [], [], [], [], [], [], []

    for name, group in progress_bar(groups, total=groups.ngroups, parent=mb, comment="Merging"):
        if len(numpy.unique(group["label"].values)) > 1:
            raise ValueError("More than one label in the group")
        area_tot.append(group["Area"].values)
        flux_tot.append(group["Flux"].values)
        losv_tot.append(group["LOS_V"].values)
        X_tot.append(group["X"].values)
        Y_tot.append(group["Y"].values)
        label_tot.append(group["label"].values)
        frame_tot.append(group["frame"].values)
        ecc_tot.append(group["ecc"].values)

    df_final = pandas.DataFrame({
        "label":    [x[0]  for x in label_tot],
        "Lifetime": [len(x) for x in frame_tot],
        "X": X_tot, "Y": Y_tot,
        "Area": area_tot, "Flux": flux_tot,
        "LOS_V": losv_tot, "Frames": frame_tot, "ecc": ecc_tot
    })
    df_final = df_final[df_final["Lifetime"] >= minliftime]

    vxtot, vytot, stdvxtot, stdvytot = [], [], [], []
    for j in progress_bar(range(len(df_final)), parent=mb, comment="Velocities"):
        x  = numpy.array(df_final["X"].iloc[j])
        y  = numpy.array(df_final["Y"].iloc[j])
        vx = _central_diff_velocity(x, dx, dt)
        vy = _central_diff_velocity(y, dx, dt)
        vxtot.append(vx)
        vytot.append(vy)
        stdvxtot.append(numpy.std(vx))
        stdvytot.append(numpy.std(vy))

    df_final["Vx"]    = vxtot
    df_final["Vy"]    = vytot
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
    for j, file in progress_bar(list(enumerate(files)), comment="Tabulation"):
        src_img = astropy.io.fits.getdata(filesB[j], memmap=False)
        asc_img = astropy.io.fits.getdata(file, memmap=False)
        unique_ids = numpy.unique(asc_img)
        for i in unique_ids:
            if i == 0:
                continue
            mask = (asc_img == i)
            Area = int(mask.sum())
            if Area == 0:
                continue

            signal = numpy.nan_to_num(
                numpy.asarray(src_img[mask], dtype=numpy.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            weights = numpy.abs(signal)
            weight_sum = weights.sum()
            if not numpy.isfinite(weight_sum) or weight_sum <= 0:
                continue

            # Signed mean flux density, but polarity-independent centroid.
            Flux = signal.sum() / Area
            X = numpy.sum(x_1[mask] * weights) / weight_sum
            Y = numpy.sum(y_1[mask] * weights) / weight_sum
            r = numpy.sqrt(Area / numpy.pi)
            circle = ((x_1 - X) ** 2 + (y_1 - Y) ** 2 < r ** 2).astype(int) * mask
            ecc = circle.sum() / Area
            temp = pandas.DataFrame([[i, X, Y, Area, Flux, j, ecc]], columns=["label", "X", "Y", "Area", "Flux", "frame", "ecc"])
            df = pandas.concat([df, temp], ignore_index=False)

    groups = df.groupby("label")
    area_tot, flux_tot, X_tot, Y_tot, label_tot, frame_tot, ecc_tot = [], [], [], [], [], [], []

    for name, group in progress_bar(list(groups), total=groups.ngroups, comment="Merging"):
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
    for j in progress_bar(range(len(df_final)), comment="Velocities"):
        x = numpy.array(df_final["X"].iloc[j])
        y = numpy.array(df_final["Y"].iloc[j])
        vx = _central_diff_velocity(x, dx, dt)
        vy = _central_diff_velocity(y, dx, dt)
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
