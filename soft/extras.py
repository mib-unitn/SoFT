"""
soft_extras.py
==============
Extended analysis tools for SoFT (Solar Feature Tracking) dataframes.

This module provides two independent sets of functionality:

Level 2 enrichment  (``build_level2``)
    Enrich a SoFT Level 1 dataframe with derived physical quantities:
      1. Trajectory angle  – dominant drift direction via EMD.
      2. Cancellation      – three closest co-dying opposite-polarity neighbours.
      3. Death cause       – mutually exclusive ``"cancelled"`` / ``"subsumed"``
                             classification with adjudication by proximity.
      4. Coronal hole flag – AIA 193 Å / HMI mask from JSOC.

Feature maps  (``build_feature_maps``)
    For features with lifetime above a threshold, paint circular blobs on a
    reference image canvas — sized by mean area, coloured by either peak
    oscillation frequency or velocity MAD — and save the result as FITS.

Quick start
-----------
    import soft_extras as sx

    # Level 2
    df_l2 = sx.build_level2(df, "2023-06-01T12:00:00",
                             jsoc_email="you@example.com")

    # Feature maps
    sx.build_feature_maps(df, img_path="data/00-data/0030.fits",
                          output_dir="data/", lifetime_threshold=15)

Dependencies
------------
    numpy, pandas, scipy, astropy, sunpy, PyEMD, tqdm, soft
"""

import gc
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from scipy.signal import periodogram
from scipy.spatial import KDTree
from sunpy.net import Fido, attrs as a
import sunpy.map
from tqdm import tqdm
from PyEMD import EMD
import soft.soft as st


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_norm(img: np.ndarray) -> np.ndarray:
    """Min-max normalise *img* → [0, 1] as float32. Returns NaN array if flat."""
    lo, hi = np.nanmin(img), np.nanmax(img)
    span = hi - lo
    if span == 0:
        return np.full_like(img, np.nan, dtype=np.float32)
    out = np.subtract(img, lo, dtype=np.float32)
    np.divide(out, span, out=out)
    return out


def _ensure_2d(
    dist: np.ndarray,
    idx:  np.ndarray,
    k:    int,
    cancellation_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad KDTree query results to shape (N, cancellation_k)."""
    if k == 1:
        dist = dist.reshape(-1, 1)
        idx  = idx.reshape(-1, 1)
    if k < cancellation_k:
        pad  = cancellation_k - k
        dist = np.pad(dist.astype(float), ((0, 0), (0, pad)), constant_values=np.nan)
        idx  = np.pad(idx.astype(float),  ((0, 0), (0, pad)), constant_values=-1)
    return dist, idx


def _emd_angle(x_traj, y_traj) -> float:
    """
    Mean trajectory angle (degrees) from EMD residual of re-centred positions.
    Returns NaN on any failure or if the trajectory has fewer than 2 points.
    """
    try:
        x = np.asarray(x_traj, dtype=float)
        y = np.asarray(y_traj, dtype=float)
        if len(x) < 2:
            return np.nan
        emd       = EMD()
        imfs_x    = emd.emd(x - x[0])
        imfs_y    = emd.emd(y - y[0])
        angle_rad = np.arctan2(imfs_y[-1], imfs_x[-1])
        return float(np.mean(angle_rad) * 180.0 / np.pi)
    except Exception:
        return np.nan


def _mean_of_series(v) -> float:
    """Reduce a time-series column (list / array) to a scalar nanmean."""
    try:
        return float(np.nanmean(v))
    except (TypeError, ValueError):
        return float(v)


def _paint_disk(
    canvas: np.ndarray,
    cx:     float,
    cy:     float,
    radius: float,
    value:  float,
) -> None:
    """
    Paint a filled circle on *canvas* in-place.

    Only pixels within the canvas bounds are written.  Overlapping blobs are
    overwritten by the later one (last-write-wins), consistent with the
    original behaviour.
    """
    r = int(np.ceil(radius))
    y0 = max(0,              int(cy) - r)
    y1 = min(canvas.shape[0], int(cy) + r + 1)
    x0 = max(0,              int(cx) - r)
    x1 = min(canvas.shape[1], int(cx) + r + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask   = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    canvas[y0:y1, x0:x1][mask] = value


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 – coronal hole mask
# ──────────────────────────────────────────────────────────────────────────────

def get_ch_mask(
    target_time: str,
    jsoc_email:  str   = "your_email@example.com",
    num_retries: int   = 1,
    l_thr:       float = 0.985,
    h_thr:       float = 0.995,
    min_distance: int  = 300,
    min_size:    int   = 500,
) -> np.ndarray | None:
    """
    Download AIA 193 Å and HMI 45 s data from JSOC, reproject AIA onto the HMI
    pixel grid, and return a binary coronal-hole mask (int8, 1 = coronal hole).

    Parameters
    ----------
    target_time  : ISO-8601 string, e.g. ``"2023-06-01T12:00:00"``.
    jsoc_email   : Registered JSOC notification address.
                   Register at http://jsoc.stanford.edu/ajax/register_email.html
    num_retries  : How many times to retry on failure, shifting +1 h each time.
    l_thr        : Lower intensity threshold for CH detection (default 0.985).
    h_thr        : Upper intensity threshold for CH detection (default 0.995).
    min_distance : Minimum distance between local maxima in pixels (default 300).
    min_size     : Minimum coronal hole size in pixels (default 500).

    Returns
    -------
    np.ndarray of shape (H, W) with dtype int8, or *None* on total failure.
    """
    t0 = datetime.fromisoformat(target_time)

    for attempt in range(num_retries + 1):
        try:
            print(f"[CH mask] Attempt {attempt}/{num_retries} — querying JSOC for {t0} …")

            hmi_res = Fido.search(
                a.Time(t0, t0 + timedelta(seconds=45)),
                a.jsoc.Series("hmi.M_45s"),
                a.jsoc.Notify(jsoc_email),
            )
            aia_res = Fido.search(
                a.Time(t0, t0 + timedelta(seconds=12)),
                a.jsoc.Series("aia.lev1_euv_12s"),
                a.Wavelength(193 * u.angstrom),
                a.jsoc.Notify(jsoc_email),
            )

            if hmi_res.file_num == 0 or aia_res.file_num == 0:
                raise ValueError("No HMI or AIA data found for this time window.")

            print("  Fetching files …")
            hmi_map = sunpy.map.Map(Fido.fetch(hmi_res[:, 0], dpath="./hmi_data/")[0])
            aia_map = sunpy.map.Map(Fido.fetch(aia_res[:, 0], dpath="./aia_data/")[0])

            print("  Reprojecting AIA → HMI grid …")
            aia_data = aia_map.reproject_to(hmi_map.wcs).data.astype(np.float32)
            del hmi_map, aia_map
            gc.collect()

            aia_inv = _safe_norm(aia_data)
            np.subtract(1.0, aia_inv, out=aia_inv)   # dark → bright
            del aia_data
            gc.collect()

            chs  = st.detection(aia_inv, l_thr=l_thr, h_thr=h_thr,
                                 min_distance=min_distance, sign="pos",
                                 separation=False)
            del aia_inv
            gc.collect()

            mask = (st.identification(chs, min_size=min_size) > 0).astype(np.int8)
            del chs
            gc.collect()

            print("  Coronal hole mask ready.")
            return mask

        except Exception as exc:
            print(f"  Attempt failed: {exc}")
            if attempt < num_retries:
                t0 += timedelta(hours=1)
                print(f"  Retrying with shifted time {t0} …\n")
            else:
                print(f"  All retries exhausted. Returning None.")
                return None


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 – trajectory angles
# ──────────────────────────────────────────────────────────────────────────────

def compute_trajectory_angles(
    df:        pd.DataFrame,
    x_col:     str = "X",
    y_col:     str = "Y",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Add ``traj_angle_deg`` — the EMD-derived mean propagation angle (degrees) —
    to every row of *df*.

    Parameters
    ----------
    df        : Must have list-valued columns *x_col* and *y_col*.
    x_col     : Column with the x-position trajectory.
    y_col     : Column with the y-position trajectory.
    n_workers : Parallel worker processes for the EMD computation.

    Returns
    -------
    df with a new ``traj_angle_deg`` column (NaN where undetermined).
    """
    args = list(zip(df[x_col].tolist(), df[y_col].tolist()))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_emd_angle, x, y): i for i, (x, y) in enumerate(args)}
        results = {}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Trajectory angles", leave=False):
            results[futures[fut]] = fut.result()

    df = df.copy()
    df["traj_angle_deg"] = [results[i] for i in range(len(args))]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 – cancellation
# ──────────────────────────────────────────────────────────────────────────────

def compute_cancellation(
    df:             pd.DataFrame,
    cancellation_k: int = 3,
) -> pd.DataFrame:
    """
    For each feature, find the *cancellation_k* closest opposite-polarity
    neighbours **that share the same last frame** (i.e. co-dying features) and
    record their labels and Euclidean distances.

    Requires columns: ``Frames``, ``label``, ``meanX``, ``meanY``.

    Adds ``n1_label … n{k}_label`` and ``n1_dist … n{k}_dist``.

    Parameters
    ----------
    df             : SoFT Level 1 dataframe.
    cancellation_k : Number of nearest opposite-polarity neighbours to store
                     (default 3).
    """
    df = df.copy()
    df["_last_frame"] = df["Frames"].apply(lambda x: x[-1])
    df["_sign"]       = np.sign(df["label"]).astype(int)

    for i in range(1, cancellation_k + 1):
        df[f"n{i}_label"] = np.nan
        df[f"n{i}_dist"]  = np.nan

    groups = df.groupby("_last_frame")

    for _, grp in tqdm(groups, total=groups.ngroups,
                       desc="Cancellation search", leave=False):
        pos    = grp[grp["_sign"] > 0]
        neg    = grp[grp["_sign"] < 0]
        if pos.empty or neg.empty:
            continue

        pos_xy = pos[["meanX", "meanY"]].values
        neg_xy = neg[["meanX", "meanY"]].values

        for (query, tree_pts, tree_labels, col_prefix) in [
            (pos_xy, neg_xy, neg["label"].values, pos.index),
            (neg_xy, pos_xy, pos["label"].values, neg.index),
        ]:
            k        = min(cancellation_k, len(tree_pts))
            d, i_arr = KDTree(tree_pts).query(query, k=k)
            d, i_arr = _ensure_2d(d, i_arr, k, cancellation_k)

            for col in range(cancellation_k):
                valid           = i_arr[:, col] != -1
                lbls            = np.full(len(query), np.nan)
                lbls[valid]     = tree_labels[i_arr[valid, col].astype(int)]
                df.loc[col_prefix, f"n{col+1}_label"] = lbls
                df.loc[col_prefix, f"n{col+1}_dist"]  = d[:, col]

    df.drop(columns=["_last_frame", "_sign"], inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 – death cause (cancellation vs subsumption)
# ──────────────────────────────────────────────────────────────────────────────

def compute_death_cause(
    df:             pd.DataFrame,
    size_col:       str   = "Area",
    max_dist:       float = 15.0,
    size_ratio:     float = 2.0,
    cancellation_k: int   = 3,
) -> pd.DataFrame:
    """
    Classify the likely cause of death for every dying feature as either
    ``"cancelled"`` or ``"subsumed"``.  The two outcomes are mutually exclusive.

    **Cancelled** — dies alongside a co-dying opposite-polarity neighbour
    (same last frame).  The closest such neighbour is taken directly from the
    ``n1_label`` / ``n1_dist`` columns produced by ``compute_cancellation``.

    **Subsumed** — dies while a *surviving* opposite-polarity feature (active
    at the dying frame but outliving it) lies within *max_dist* pixels and is
    at least *size_ratio* × larger.  This pool is queried with a separate
    KD-tree that is entirely distinct from the co-dying pool.

    **Adjudication when both candidates exist**: the closer one wins.  If the
    surviving host is closer than the co-dying partner, the interpretation is
    that both small features were ingested by the large one.

    Requires ``compute_cancellation`` to have been run first.

    Parameters
    ----------
    df         : Post-cancellation Level 2 dataframe.
    size_col   : Area time-series column; the mean is used as the scalar size.
    max_dist   : Pixel distance cap for subsumption host (default 15).
    size_ratio : Minimum host / candidate size ratio (default 2).

    Returns
    -------
    df with five new columns: ``cancelled``, ``cancelled_by``, ``subsumed``,
    ``subsumed_by``, ``death_cause``.
    """
    required = [f"n{i}{s}" for i in range(1, cancellation_k + 1)
                            for s in ("_label", "_dist")]
    if missing := [c for c in required if c not in df.columns]:
        raise ValueError(
            f"compute_death_cause requires cancellation columns {missing}. "
            "Run compute_cancellation first."
        )
    if size_col not in df.columns:
        raise ValueError(f"Size column '{size_col}' not found.")

    df = df.copy()
    df["_mean_size"]   = df[size_col].apply(_mean_of_series)
    df["_first_frame"] = df["Frames"].apply(lambda x: x[0])
    df["_last_frame"]  = df["Frames"].apply(lambda x: x[-1])
    df["_sign"]        = np.sign(df["label"]).astype(int)

    global_last       = df["_last_frame"].max()
    label_to_size     = df.set_index("label")["_mean_size"].to_dict()
    iloc_map          = {idx: pos for pos, idx in enumerate(df.index)}

    # Output arrays
    cancelled    = np.zeros(len(df), dtype=bool)
    cancelled_by = np.full(len(df), np.nan)
    subsumed     = np.zeros(len(df), dtype=bool)
    subsumed_by  = np.full(len(df), np.nan)
    _can_dist    = np.full(len(df), np.inf)
    _sub_dist    = np.full(len(df), np.inf)

    dying = df[df["_last_frame"] < global_last]

    # ── Pass 1: cancellation — read directly from n1 columns ─────────────────
    for idx in dying.index:
        lbl  = df.at[idx, "n1_label"]
        dist = df.at[idx, "n1_dist"]
        if not (np.isnan(lbl) or np.isnan(dist)):
            pos              = iloc_map[idx]
            cancelled[pos]   = True
            cancelled_by[pos] = lbl
            _can_dist[pos]   = dist

    # ── Pass 2: subsumption — query surviving features per dying-frame group ──
    for dying_frame, dying_grp in tqdm(
        dying.groupby("_last_frame"),
        total=dying["_last_frame"].nunique(),
        desc="Death cause (subsumption)",
        leave=False,
    ):
        surviving = df[
            (df["_first_frame"] <= dying_frame) &
            (df["_last_frame"]  >  dying_frame)
        ]
        if surviving.empty:
            continue

        for dying_sub, opp_surv in [
            (dying_grp[dying_grp["_sign"] > 0], surviving[surviving["_sign"] < 0]),
            (dying_grp[dying_grp["_sign"] < 0], surviving[surviving["_sign"] > 0]),
        ]:
            if dying_sub.empty or opp_surv.empty:
                continue

            surv_xy      = opp_surv[["meanX", "meanY"]].values
            dying_xy     = dying_sub[["meanX", "meanY"]].values
            nearby_lists = KDTree(surv_xy).query_ball_point(dying_xy, r=max_dist)

            for local_i, (idx, nearby) in enumerate(
                zip(dying_sub.index, nearby_lists)
            ):
                if not nearby:
                    continue
                own_size = df.at[idx, "_mean_size"]
                if own_size == 0 or np.isnan(own_size):
                    continue

                best_dist, best_label = np.inf, np.nan
                for ni in nearby:
                    nb       = opp_surv.iloc[ni]
                    nb_size  = nb["_mean_size"]
                    if np.isnan(nb_size) or nb_size < size_ratio * own_size:
                        continue
                    d = float(np.hypot(
                        dying_xy[local_i, 0] - nb["meanX"],
                        dying_xy[local_i, 1] - nb["meanY"],
                    ))
                    if d < best_dist:
                        best_dist, best_label = d, nb["label"]

                if not np.isnan(best_label):
                    pos             = iloc_map[idx]
                    subsumed[pos]   = True
                    subsumed_by[pos] = best_label
                    _sub_dist[pos]  = best_dist

    # ── Adjudication: enforce mutual exclusivity (closer candidate wins) ──────
    both      = cancelled & subsumed
    sub_wins  = both & (_sub_dist <= _can_dist)
    can_wins  = both & (_sub_dist >  _can_dist)

    cancelled[sub_wins]     = False
    cancelled_by[sub_wins]  = np.nan
    subsumed[can_wins]      = False
    subsumed_by[can_wins]   = np.nan

    death_cause             = np.full(len(df), np.nan, dtype=object)
    death_cause[cancelled]  = "cancelled"
    death_cause[subsumed]   = "subsumed"

    df["cancelled"]    = cancelled
    df["cancelled_by"] = cancelled_by
    df["subsumed"]     = subsumed
    df["subsumed_by"]  = subsumed_by
    df["death_cause"]  = death_cause

    df.drop(columns=["_mean_size", "_first_frame", "_last_frame", "_sign"],
            inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 – coronal hole flag
# ──────────────────────────────────────────────────────────────────────────────

def flag_coronal_hole(
    df:    pd.DataFrame,
    mask:  np.ndarray,
    x_col: str = "meanX",
    y_col: str = "meanY",
) -> pd.DataFrame:
    """
    Add ``in_coronal_hole`` (bool) — True if the feature's mean position falls
    inside the coronal hole mask.

    Parameters
    ----------
    df   : Feature dataframe with pixel-space position columns.
    mask : 2-D int8 array (1 = coronal hole) aligned with the pixel grid.
    """
    df   = df.copy()
    rows = np.clip(df[y_col].values.round().astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(df[x_col].values.round().astype(int), 0, mask.shape[1] - 1)
    df["in_coronal_hole"] = mask[rows, cols].astype(bool)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 – master pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_level2(
    df:                   pd.DataFrame,
    target_time:          str,
    # ── step toggles (all on by default) ─────────────────────────────────────
    run_angles:           bool  = True,
    run_cancellation:     bool  = True,
    run_death_cause:      bool  = True,
    run_coronal_hole:     bool  = True,
    # ── column names ─────────────────────────────────────────────────────────
    x_traj_col:           str   = "X",
    y_traj_col:           str   = "Y",
    x_pos_col:            str   = "meanX",
    y_pos_col:            str   = "meanY",
    size_col:             str   = "Area",
    # ── trajectory angles ────────────────────────────────────────────────────
    angle_workers:        int   = 4,
    # ── cancellation ─────────────────────────────────────────────────────────
    cancellation_k:       int   = 3,
    # ── death cause ──────────────────────────────────────────────────────────
    subsumption_max_dist: float = 15.0,
    subsumption_ratio:    float = 2.0,
    # ── coronal hole ─────────────────────────────────────────────────────────
    jsoc_email:           str   = "your_email@example.com",
    ch_retries:           int   = 1,
    ch_l_thr:             float = 0.985,
    ch_h_thr:             float = 0.995,
    ch_min_distance:      int   = 300,
    ch_min_size:          int   = 500,
) -> pd.DataFrame:
    """
    Enrich a SoFT Level 1 dataframe with Level 2 derived quantities.

    Each step can be toggled independently.  Disabled steps still add their
    output columns filled with NaN so the schema is always consistent.

    Steps & output columns
    ----------------------
    1. **Trajectory angles** (``run_angles``)
         ``traj_angle_deg``

    2. **Cancellation** (``run_cancellation``)
         ``n1…n3_label``, ``n1…n3_dist``

    3. **Death cause** (``run_death_cause``, needs cancellation)
         ``cancelled``, ``cancelled_by``, ``subsumed``, ``subsumed_by``,
         ``death_cause``

    4. **Coronal hole** (``run_coronal_hole``)
         ``in_coronal_hole``

    Parameters
    ----------
    df                   : SoFT Level 1 dataframe.
    target_time          : ISO-8601 time for the coronal hole mask download.
    run_angles           : Toggle step 1 (default True).
    run_cancellation     : Toggle step 2 (default True).
    run_death_cause      : Toggle step 3 (default True).
    run_coronal_hole     : Toggle step 4 (default True).
    x_traj_col           : X-trajectory column name (default ``"X"``).
    y_traj_col           : Y-trajectory column name (default ``"Y"``).
    x_pos_col            : Mean x position column (default ``"meanX"``).
    y_pos_col            : Mean y position column (default ``"meanY"``).
    size_col             : Area time-series column (default ``"Area"``).
    angle_workers        : Parallel workers for EMD angle computation (default 4).
    cancellation_k       : Nearest co-dying neighbours to store (default 3).
    subsumption_max_dist : Max pixel distance to subsumption host (default 15).
    subsumption_ratio    : Min host/candidate size ratio (default 2).
    jsoc_email           : Registered JSOC email.
    ch_retries           : Retry budget for the JSOC download (default 1).
    ch_l_thr             : Lower threshold for CH detection (default 0.985).
    ch_h_thr             : Upper threshold for CH detection (default 0.995).
    ch_min_distance      : Min distance between CH maxima in pixels (default 300).
    ch_min_size          : Min coronal hole size in pixels (default 500).

    Examples
    --------
    Run all steps:

    >>> df_l2 = build_level2(df, "2023-06-01T12:00:00")

    Offline run — skip JSOC:

    >>> df_l2 = build_level2(df, "2023-06-01T12:00:00", run_coronal_hole=False)

    Only cancellation + death cause:

    >>> df_l2 = build_level2(df, "2023-06-01T12:00:00",
    ...                      run_angles=False, run_coronal_hole=False)
    """
    df_l2       = df.copy()
    total_steps = sum([run_angles, run_cancellation, run_death_cause, run_coronal_hole])
    step        = 0

    def _header(label: str) -> str:
        nonlocal step
        step += 1
        return f"\n[Level 2]  Step {step}/{total_steps} — {label} …"

    # ── 1. Trajectory angles ──────────────────────────────────────────────────
    if run_angles:
        print(_header("Trajectory angles"))
        if x_traj_col in df_l2.columns and y_traj_col in df_l2.columns:
            df_l2 = compute_trajectory_angles(df_l2, x_col=x_traj_col,
                                               y_col=y_traj_col,
                                               n_workers=angle_workers)
        else:
            warnings.warn(f"Columns '{x_traj_col}'/'{y_traj_col}' not found; "
                          "skipping trajectory angles.", stacklevel=2)
            df_l2["traj_angle_deg"] = np.nan
    else:
        print("[Level 2]  Trajectory angles … SKIPPED")
        df_l2["traj_angle_deg"] = np.nan

    # ── 2. Cancellation ───────────────────────────────────────────────────────
    if run_cancellation:
        print(_header("Cancellation"))
        df_l2 = compute_cancellation(df_l2, cancellation_k=cancellation_k)
    else:
        print("[Level 2]  Cancellation … SKIPPED")
        for i in range(1, cancellation_k + 1):
            df_l2[f"n{i}_label"] = np.nan
            df_l2[f"n{i}_dist"]  = np.nan

    # ── 3. Death cause ────────────────────────────────────────────────────────
    _null_dc = dict(cancelled=np.nan, cancelled_by=np.nan,
                    subsumed=np.nan,  subsumed_by=np.nan,
                    death_cause=np.nan)
    if run_death_cause:
        if not run_cancellation:
            warnings.warn("Death cause requires cancellation columns but "
                          "run_cancellation=False; skipping.", stacklevel=2)
            df_l2 = df_l2.assign(**_null_dc)
        elif size_col not in df_l2.columns:
            warnings.warn(f"Column '{size_col}' not found; "
                          "skipping death cause.", stacklevel=2)
            df_l2 = df_l2.assign(**_null_dc)
        else:
            print(_header("Death cause"))
            df_l2 = compute_death_cause(df_l2, size_col=size_col,
                                         max_dist=subsumption_max_dist,
                                         size_ratio=subsumption_ratio,
                                         cancellation_k=cancellation_k)
    else:
        print("[Level 2]  Death cause … SKIPPED")
        df_l2 = df_l2.assign(**_null_dc)

    # ── 4. Coronal hole ───────────────────────────────────────────────────────
    if run_coronal_hole:
        print(_header("Coronal hole detection"))
        mask = get_ch_mask(target_time, jsoc_email=jsoc_email,
                           num_retries=ch_retries, l_thr=ch_l_thr,
                           h_thr=ch_h_thr, min_distance=ch_min_distance,
                           min_size=ch_min_size)
        if mask is not None:
            df_l2 = flag_coronal_hole(df_l2, mask,
                                      x_col=x_pos_col, y_col=y_pos_col)
        else:
            warnings.warn("CH mask unavailable; 'in_coronal_hole' = NaN.",
                          stacklevel=2)
            df_l2["in_coronal_hole"] = np.nan
    else:
        print("[Level 2]  Coronal hole detection … SKIPPED")
        df_l2["in_coronal_hole"] = np.nan

    print("\n[Level 2]  Done.\n")
    return df_l2


# ──────────────────────────────────────────────────────────────────────────────
# Feature maps
# ──────────────────────────────────────────────────────────────────────────────

def _feature_spectral_stats(
    row,
    fs:       float = 1 / 45.0,
    nfft:     int   = 256,
    freq_min: float = 1.5,
) -> dict:
    """
    Compute spectral and variability statistics for a single feature row.

    Returns a dict with keys: ``meanX``, ``meanY``, ``mean_area``,
    ``peak_freq_mhz``, ``velocity_mad``.
    """
    vx = np.asarray(row.Vx, dtype=float)
    vy = np.asarray(row.Vy, dtype=float)

    f, pxx = periodogram(vx, fs=fs, nfft=nfft,
                         detrend="linear", scaling="density", window="tukey")
    _, pyy = periodogram(vy, fs=fs, nfft=nfft,
                         detrend="linear", scaling="density", window="tukey")

    # Normalise to relative power and convert frequency to mHz
    pxx = pxx / np.sum(pxx)
    pyy = pyy / np.sum(pyy)
    f   = f * 1e3

    # Discard low-frequency drift below freq_min mHz
    mask    = f > freq_min
    f_cut   = f[mask]
    pxx_cut = pxx[mask]
    pyy_cut = pyy[mask]

    peak_freq = (f_cut[np.argmax(pxx_cut)] + f_cut[np.argmax(pyy_cut)]) / 2.0

    mad_vx = np.median(np.abs(vx - np.median(vx)))
    mad_vy = np.median(np.abs(vy - np.median(vy)))

    return {
        "meanX":         float(np.mean(row.X)),
        "meanY":         float(np.mean(row.Y)),
        "mean_area":     float(np.nanmean(row.Area)),
        "peak_freq_mhz": float(peak_freq),
        "velocity_mad":  float((mad_vx + mad_vy) / 2.0),
    }


def build_feature_maps(
    df:                 pd.DataFrame,
    img_path:           str,
    output_dir:         str  = ".",
    lifetime_threshold: int  = 15,
    cadence_s:          float = 45.0,
    nfft:               int  = 256,
    freq_min_mhz:       float = 1.5,
    freq_map_name:      str  = "peak_freq_map.fits",
    mad_map_name:       str  = "velocity_mad_map.fits",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build two FITS maps from features with ``Lifetime > lifetime_threshold``:

    * **Peak frequency map** — each feature is painted as a filled circle sized
      by its mean area and valued by its peak oscillation frequency (mHz).
    * **Velocity MAD map**   — same geometry, valued by the mean median absolute
      deviation of Vx and Vy.

    Both maps are saved to *output_dir* as FITS files carrying the WCS header
    from *img_path* so that solar coordinates are preserved.

    Parameters
    ----------
    df                 : SoFT dataframe with columns ``Lifetime``, ``X``, ``Y``,
                         ``Area``, ``Vx``, ``Vy``.
    img_path           : Path to a reference FITS frame (defines the pixel grid
                         and WCS header to embed in the output).
    output_dir         : Directory for output FITS files (default: current dir).
    lifetime_threshold : Minimum lifetime in frames (default 15).
    cadence_s          : Instrument cadence in seconds used as 1/fs (default 45).
    nfft               : FFT length for the periodogram (default 256).
    freq_min_mhz       : Low-frequency cutoff in mHz to suppress drift (default 1.5).
    freq_map_name      : Output filename for the frequency map (default
                         ``"peak_freq_map.fits"``).
    mad_map_name       : Output filename for the MAD map (default
                         ``"velocity_mad_map.fits"``).

    Returns
    -------
    (freq_map, mad_map) — the two 2-D float64 arrays before saving.

    Notes
    -----
    Overlapping blobs are overwritten last-write-wins, consistent with the
    expected sparsity of long-lived features.  Zero pixels in the output
    indicate regions with no qualifying feature.
    """
    img    = fits.getdata(img_path).astype(float)
    header = fits.getheader(img_path)

    long_lived = df[df["Lifetime"] > lifetime_threshold]
    if long_lived.empty:
        warnings.warn(
            f"No features with Lifetime > {lifetime_threshold}. "
            "Returning zero maps.", stacklevel=2
        )
        zero = np.zeros_like(img)
        fits.writeto(os.path.join(output_dir, freq_map_name), zero,
                     header=header, overwrite=True)
        fits.writeto(os.path.join(output_dir, mad_map_name),  zero,
                     header=header, overwrite=True)
        return zero, zero

    fs = 1.0 / cadence_s

    freq_map = np.zeros_like(img)
    mad_map  = np.zeros_like(img)

    for row in tqdm(long_lived.itertuples(), total=len(long_lived),
                    desc="Building feature maps", leave=False):
        stats  = _feature_spectral_stats(row, fs=fs, nfft=nfft,
                                         freq_min=freq_min_mhz)
        radius = np.sqrt(stats["mean_area"] / np.pi)

        _paint_disk(freq_map, stats["meanX"], stats["meanY"],
                    radius, stats["peak_freq_mhz"])
        _paint_disk(mad_map,  stats["meanX"], stats["meanY"],
                    radius, stats["velocity_mad"])

    os.makedirs(output_dir, exist_ok=True)
    fits.writeto(os.path.join(output_dir, freq_map_name),
                 freq_map, header=header, overwrite=True)
    fits.writeto(os.path.join(output_dir, mad_map_name),
                 mad_map,  header=header, overwrite=True)

    print(f"Saved: {os.path.join(output_dir, freq_map_name)}")
    print(f"Saved: {os.path.join(output_dir, mad_map_name)}")

    return freq_map, mad_map
