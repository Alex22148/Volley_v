import time
import numpy as np
import itertools
import json

WORK = False
if WORK:
    import torch

CAMERAS = ["cL", "cR", "xL", "xR"]

configuration_est3D = {
        "N": 16,  # ilosć zestawów w pakiecie
        "M": 4,  # ilość kamer
        "O": 5,  # max. ilość analizowanych koordybat obrazowych po inferencji
        "bounds": ((-16, 16), (-8, 8), (-1, 10)),  # granice sceny 3D
        "min_conf": 0.5,  # próg akceptacji koordynat 2D z inferencji
        "min_angle_deg": 1.0,
        "max_err": 25,
    }



# ============================================================
# 🧠 Szybka funkcja undistorsji
# ============================================================
def undistort_points_numpy_iterative(pts, K_all, dist_all, max_iter=7):
    N, M, O, _ = pts.shape
    pts_out = np.empty_like(pts, dtype=np.float32)
    for m in range(M):
        fx, fy = K_all[m,0,0], K_all[m,1,1]
        cx, cy = K_all[m,0,2], K_all[m,1,2]
        k1 = dist_all[m]
        pts_m = pts[:,m,:,:]
        x_d = (pts_m[...,0] - cx)/fx
        y_d = (pts_m[...,1] - cy)/fy
        x_u, y_u = x_d.copy(), y_d.copy()
        for _ in range(max_iter):
            r2 = x_u*x_u + y_u*y_u
            scale = 1.0 + k1*r2
            x_u = x_d/scale
            y_u = y_d/scale
        pts_out[:,m,:,0] = x_u*fx + cx
        pts_out[:,m,:,1] = y_u*fy + cy
    return pts_out



# def triangulate_all_variants(pts_undist, P_all):
#     return None, None

def triangulate_all_variants(pts_undist, P_all):
    """
    Wyznacz wszystkie możliwe punkty 3D dla par kamer i kandydatów 2D.

    Parameters
    ----------
    pts_undist : ndarray, shape (N, M, O, 2)
        Punkty 2D po undistort (N=liczba batchy, M=liczba kamer, O=max liczba kandydatów).
    P_all : ndarray, shape (M, 3, 4)
        Macierze projekcji dla wszystkich kamer.

    Returns
    -------
    X_all : ndarray, shape (N, V, 3)
        Wszystkie możliwe punkty 3D (NaN jeśli nie można wyliczyć).
    meta_all : ndarray, shape (N, V, 4)
        Opis wariantów: (camA, camB, idxA, idxB).
    """
    N, M, O, _ = pts_undist.shape
    variants = list(itertools.combinations(range(M), 2))  # wszystkie pary kamer
    V = len(variants) * O * O  # liczba wariantów per batch

    X_all = np.full((N, V, 3), np.nan, dtype=np.float32)
    meta_all = np.full((N, V, 4), -1, dtype=np.int32)

    for n in range(N):
        v_idx = 0
        for camA, camB in variants:
            ptsA = pts_undist[n, camA]  # shape (O,2)
            ptsB = pts_undist[n, camB]  # shape (O,2)

            for idxA in range(O):
                xA, yA = ptsA[idxA]
                if np.isnan(xA) or np.isnan(yA):
                    v_idx += O
                    continue

                for idxB in range(O):
                    xB, yB = ptsB[idxB]
                    if np.isnan(xB) or np.isnan(yB):
                        v_idx += 1
                        continue

                    # budowa macierzy A
                    PA, PB = P_all[camA], P_all[camB]
                    A = np.array([
                        xA * PA[2] - PA[0],
                        yA * PA[2] - PA[1],
                        xB * PB[2] - PB[0],
                        yB * PB[2] - PB[1],
                    ])

                    # SVD
                    try:
                        _, _, Vt = np.linalg.svd(A)
                        X_h = Vt[-1]
                        X = X_h[:3] / X_h[3]
                        X_all[n, v_idx] = X
                        meta_all[n, v_idx] = [camA, camB, idxA, idxB]
                    except np.linalg.LinAlgError:
                        pass

                    v_idx += 1

    return X_all, meta_all



def analyze_triangulated_points_vec(X_all, meta, pts_undist, P_all, C_all,
                                    min_angle_deg=1.0, bounds=None):
    """
    Analiza wyników triangulacji.

    Parametry:
    -----------
    X_all : np.ndarray
        shape (N, V, 3) – punkty 3D (mogą zawierać NaN).
    meta : np.ndarray
        shape (N, V, 4), gdzie każda kolumna to [camA, camB, idxA, idxB].
    pts_undist : np.ndarray
        shape (N, M, O, 2) – punkty 2D po undistorsji.
    P_all : np.ndarray
        shape (M, 3, 4) – macierze projekcji dla kamer.
    C_all : np.ndarray
        shape (M, 3) – centra kamer.
    min_angle_thresh : float
        Minimalny kąt w stopniach.
    bounds : tuple | None
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) – ograniczenia przestrzenne.

    Zwraca:
    --------
    list(dict) – lista słowników z analizą dla każdego poprawnego punktu 3D.
    """

    analyzed = []
    N, V, _ = X_all.shape


    for n in range(N):
        Xn = X_all[n]          # (V,3)
        metan = meta[n]        # (V,4)
        mask = ~np.isnan(Xn).any(axis=1)
        if not np.any(mask):
            continue

        Xn = Xn[mask]
        metan = metan[mask]

        # --- reprojekcja i błąd ---
        Xh = np.hstack([Xn, np.ones((Xn.shape[0], 1))])  # (V,4)
        proj = (P_all @ Xh.T).transpose(0, 2, 1)         # (M,V,3)
        proj = proj[..., :2] / proj[..., 2:3]

        min_errs = []
        checks = []
        angles = []

        for v in range(Xn.shape[0]):
            camA, camB, idxA, idxB = metan[v]

            # punkty 2D z kamer
            ptsA = pts_undist[n, camA, idxA]
            ptsB = pts_undist[n, camB, idxB]

            # projekcje 2D
            projA = proj[camA, v]
            projB = proj[camB, v]

            # min_err
            errA = np.linalg.norm(projA - ptsA)
            errB = np.linalg.norm(projB - ptsB)
            min_errs.append((errA + errB) / 2)

            # cheirality check
            vecA = Xn[v] - C_all[camA]
            vecB = Xn[v] - C_all[camB]
            checks.append(vecA @ (P_all[camA, 2, :3]) > 0 and
                          vecB @ (P_all[camB, 2, :3]) > 0)

            # min_angle_deg
            dot = np.dot(vecA, vecB) / (
                np.linalg.norm(vecA) * np.linalg.norm(vecB) + 1e-12
            )
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))
            angles.append(angle)

        min_errs = np.array(min_errs)
        checks = np.array(checks)
        angles = np.array(angles)

        # --- filtracja ---
        valid = (checks) & (angles > min_angle_deg)
        if bounds is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
            xb, yb, zb = Xn[:, 0], Xn[:, 1], Xn[:, 2]
            in_bounds = (xb >= xmin) & (xb <= xmax) & \
                        (yb >= ymin) & (yb <= ymax) & \
                        (zb >= zmin) & (zb <= zmax)
            valid &= in_bounds

        # print(valid)

        for v in np.where(valid)[0]:
            camA, camB, idxA, idxB = metan[v]
            record = {
                "n": int(n+1),
                "point": Xn[v].tolist(),
                "cameras": (int(camA), int(camB)),
                "indices": (int(idxA), int(idxB)),
                "min_err": float(min_errs[v]),
                "check_cheirality": bool(checks[v]),
                "min_angle_deg": float(angles[v]),
                # "in_boinds": bool(in_bounds)
            }
            analyzed.append(record)

    return analyzed


def analyze_triangulated_points_B(X_all, conf_bat, meta, pts_undist, P_all, C_all,
                                    min_angle_deg=1.0, bounds=None, max_err = 25):
    """
    Analiza wyników triangulacji.

    Parametry:
    -----------
    X_all : np.ndarray
        shape (N, V, 3) – punkty 3D (mogą zawierać NaN).
    meta : np.ndarray
        shape (N, V, 4), gdzie każda kolumna to [camA, camB, idxA, idxB].
    pts_undist : np.ndarray
        shape (N, M, O, 2) – punkty 2D po undistorsji.
    P_all : np.ndarray
        shape (M, 3, 4) – macierze projekcji dla kamer.
    C_all : np.ndarray
        shape (M, 3) – centra kamer.
    min_angle_thresh : float
        Minimalny kąt w stopniach.
    bounds : tuple | None
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) – ograniczenia przestrzenne.

    Zwraca:
    --------
    list(dict) – lista słowników z analizą dla każdego poprawnego punktu 3D.
    """

    analyzed = []
    N, V, _ = X_all.shape
    # print(X_all.shape)
    # print(conf_bat.shape)
    # print(conf_bat[0])

    for n in range(N):
        # print("...")
        # print(f"n: {n+1}")

        Xn = X_all[n]          # (V,3)
        metan = meta[n]        # (V,4)
        conf_bat_n = conf_bat[n]
        mask = ~np.isnan(Xn).any(axis=1)
        if not np.any(mask):
            continue

        Xn = Xn[mask]
        metan = metan[mask]

        # # print(mask)
        # print(conf_bat_n)

        # --- reprojekcja i błąd ---
        Xh = np.hstack([Xn, np.ones((Xn.shape[0], 1))])  # (V,4)
        proj = (P_all @ Xh.T).transpose(0, 2, 1)         # (M,V,3)
        proj = proj[..., :2] / proj[..., 2:3]

        min_errs = []
        checks = []
        angles = []
        conf = []

        for v in range(Xn.shape[0]):
            camA, camB, idxA, idxB = metan[v]
            # print("---")
            # print(metan[v])
            conf.append([float(conf_bat_n[metan[v][0]][metan[v][2]]),float(conf_bat_n[metan[v][1]][metan[v][2]])])
            # print(conf[-1])
            # punkty 2D z kamer
            ptsA = pts_undist[n, camA, idxA]
            ptsB = pts_undist[n, camB, idxB]

            # projekcje 2D
            projA = proj[camA, v]
            projB = proj[camB, v]

            # min_err
            errA = np.linalg.norm(projA - ptsA)
            errB = np.linalg.norm(projB - ptsB)
            min_errs.append((errA + errB) / 2)
            # print(f" min_errs: {min_errs[-1]}")

            # # cheirality check auto Z
            # # --- Dekompresja macierzy projekcji ---
            # _, RA, CA = decompose_projection(P_all[camA])
            # _, RB, CB = decompose_projection(P_all[camB])
            # # --- Oblicz głębokości dla punktu X ---
            # ZA = depth_in_camera(RA, CA, Xn[v])
            # ZB = depth_in_camera(RB, CB, Xn[v])
            # z_samples = [ZA, ZB]
            # median_sign = np.sign(np.median(z_samples))
            # checks_new = False
            # if median_sign >= 0: checks_new = (ZA > 0) and (ZB > 0)
            # else: checks_new = (ZA < 0) and (ZB < 0)
            # print(f" checks_new: {checks_new}")
            #
            # okA, dA, CA, fA, angA = cheirality_per_camera(P_all[camA], Xn[v], eps=1e-3)
            # okB, dB, CB, fB, angB = cheirality_per_camera(P_all[camB], Xn[v], eps=1e-3)
            # checks_new2 = okA and okB
            # print(f" checks_new2: {checks_new2}  {okA}  {okB}")


            vecA = Xn[v] - C_all[camA]
            vecB = Xn[v] - C_all[camB]

            # cheirality check
            # checks.append(vecA @ (P_all[camA, 2, :3]) > 0 and
            #               vecB @ (P_all[camB, 2, :3]) > 0)
            # print(f" checks: {checks[-1]} Xn[v]: {Xn[v]}  ---> vecA: {vecA}; vecB: {vecB}")
            # print(f"     PA: {P_all[camA, 2, :3]} PB: {P_all[camB, 2, :3]}")
            # print(f"     XA: {vecA @ (P_all[camA, 2, :3]) > 0} XB: {vecB @ (P_all[camB, 2, :3]) > 0}")
            checks.append(True)


            # min_angle_deg
            dot = np.dot(vecA, vecB) / (
                np.linalg.norm(vecA) * np.linalg.norm(vecB) + 1e-12
            )
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))
            angles.append(angle)
            # print(f" {metan[v]} --> min_errs: {min_errs[-1]}; angles: {angles[-1]}")

        min_errs = np.array(min_errs)
        checks = np.array(checks)
        angles = np.array(angles)

        # --- filtracja ---
        valid = (checks) & (angles > min_angle_deg) & (min_errs <= max_err)
        # in_bounds = False
        if bounds is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
            xb, yb, zb = Xn[:, 0], Xn[:, 1], Xn[:, 2]
            in_bounds = (xb >= xmin) & (xb <= xmax) & \
                        (yb >= ymin) & (yb <= ymax) & \
                        (zb >= zmin) & (zb <= zmax)
            valid &= in_bounds

        for v in np.where(valid)[0]:
            camA, camB, idxA, idxB = metan[v]
            record = {
                "n": int(n+1),
                "point": Xn[v].tolist(),
                "cameras": (int(camA), int(camB)),
                "indices": (int(idxA), int(idxB)),
                "min_err": float(min_errs[v]),
                "check_cheirality": bool(checks[v]),
                "min_angle_deg": float(angles[v]),
                "in_bounds": bool(in_bounds[v]),
                "conf": conf[v],
            }
            analyzed.append(record)


    return analyzed



def cheirality_per_camera(P, X, eps=1e-6):
    """
    Zwraca: (is_in_front, depth, center, forward_axis, angle_deg)
    gdzie depth = (X - C) · f, f = R^T e3 (oś +Z kamery w świecie).
    """
    _, R, C = decompose_P(P)
    f = camera_forward_axis_world(R)       # wektor 'do przodu' w świecie
    v = X - C                              # wektor od kamery do punktu
    depth = float(v @ f)                   # skalarna głębokość wzdłuż osi optycznej
    # kąt między osią a kierunkiem do punktu (diagnostyka)
    cosang = float(np.clip((v @ f)/(np.linalg.norm(v)*1.0), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosang)))
    return (depth > eps), depth, C, f, angle_deg

def decompose_P(P):
    """Zwraca K, R, C. P = K [R | t], det(R)=+1, diag(K)>0."""
    M = P[:, :3]
    K, R = _rq_3x3(M)
    # uporządkuj znaki K i R
    for i in range(3):
        if K[i,i] < 0:
            K[:,i] *= -1
            R[i,:] *= -1
    if np.linalg.det(R) < 0:
        K[:,2] *= -1
        R[2,:] *= -1
    C = _camera_center(P)
    return K, R, C

# --- RQ 3x3 (bez SciPy) ---
def _rq_3x3(M):
    def flip(m): return np.flipud(np.fliplr(m))
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = flip(R.T); Q = flip(Q.T)
    D = np.diag(np.sign(np.diag(R) + (np.diag(R)==0)))
    return R @ D, D @ Q  # K, R

def _camera_center(P):
    _,_,Vt = np.linalg.svd(P)
    c = Vt[-1];  return c[:3]/c[3]

def camera_forward_axis_world(R):
    """
    Kierunek osi optycznej kamery w układzie świata.
    Konsekwentnie definiujemy 'forward' jako światowy obraz osi +Z kamery:
      f = R^T * [0,0,1].
    Wtedy 'głębokość' punktu X względem kamery to d = (X - C) · f
    i cheirality: d > 0.
    """
    f = R.T @ np.array([0.,0.,1.])
    n = np.linalg.norm(f)
    return f / n if n > 0 else f









def camera_center_from_P(P):
    """Wyznacza centrum kamery (C) z macierzy projekcji P."""
    _, _, Vt = np.linalg.svd(P)
    c = Vt[-1]
    return c[:3] / c[3]

def decompose_projection(P):
    """Rozkłada macierz projekcji P na K, R, C."""
    M, p4 = P[:, :3], P[:, 3]
    K, R = rq_decomposition(M)
    # Korekta znaków
    for i in range(3):
        if K[i, i] < 0:
            K[:, i] *= -1
            R[i, :] *= -1
    if np.linalg.det(R) < 0:
        K[:, 2] *= -1
        R[2, :] *= -1
    C = camera_center_from_P(P)
    return K, R, C

def depth_in_camera(R, C, X):
    """Zwraca współrzędną Z punktu w układzie kamery."""
    return float(R[2] @ (X - C))

def rq_decomposition(M):
    """Wersja RQ dla macierzy 3x3 (bez SciPy)."""
    def flip(m): return np.flipud(np.fliplr(m))
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = flip(R.T)
    Q = flip(Q.T)
    # Normalizacja znaków (diag K dodatnia)
    D = np.diag(np.sign(np.diag(R) + (np.diag(R)==0)))
    R = R @ D
    Q = D @ Q
    return R, Q  # R=K, Q=R




# def estiamte3D_offline_light(pts_bat,configuration,P_all, K_all, dist_all, C_all):
#     val = []
#
#     N = configuration["N"]
#     M = configuration["M"]
#     O = configuration["O"]
#     min_conf = configuration["min_conf"]
#     bounds = configuration["bounds"]
#     min_angle_deg = configuration["min_angle_deg"]
#     max_err = configuration["max_err"]
#
#
#     t2 = time.perf_counter()
#     # pts_undist = undistort_points_numpy(pts_bat, K_all, dist_all)
#     pts_undist = undistort_points_numpy_iterative(pts_bat, K_all, dist_all, max_iter=7)
#
#
#     t3 = time.perf_counter()
#     X_all, meta = triangulate_all_variants(pts_undist, P_all)
#
#     t4 = time.perf_counter()
#     analyzed = analyze_triangulated_points_B(X_all, meta, pts_undist, P_all, C_all, min_angle_deg=min_angle_deg,
#                                                bounds=bounds)
#     t5 = time.perf_counter()
#
#     analyzed_3D = {
#         "analysed": analyzed,
#     }
#
#     trangulate = {
#         "undistort_2D": pts_undist,
#         "all_3D": X_all,
#         "meta": meta,
#     }
#
#     info_time = {
#         "Undistort": 1000 * (t3 - t2),
#         "get3D": 1000 * (t4 - t3),
#         "analyzed_3D": 1000 * (t5 - t4),
#     }
#
#     info = {
#         "trangulate": trangulate,
#         "info_time": info_time,
#         "analyzed_3D": analyzed
#     }
#
#     val = pts_undist.copy()
#     return val, info



# ===================================================================
# ===================================================================

# ==========================================================
# Wersja zoptymalizowana (jeden transfer CPU<->GPU)
def extract_results_to_matrices_batch(results, N, M, O=5, min_conf=0.5):
    pts_all = np.full((N, M, O, 2), np.nan, dtype=np.float32)
    conf_all = np.full((N, M, O), np.nan, dtype=np.float32)

    xywh_list, conf_list, counts = [], [], []
    for r in results:
        if r.boxes.xywh.shape[0] == 0:
            counts.append(0)
            xywh_list.append(torch.empty((0, 2), device="cuda", dtype=torch.float32))
            conf_list.append(torch.empty((0,), device="cuda", dtype=torch.float32))
        else:
            xywh_list.append(r.boxes.xywh[:, :2].float().to("cuda"))
            conf_list.append(r.boxes.conf.float().to("cuda"))
            counts.append(r.boxes.xywh.shape[0])

    if len(xywh_list) == 0:
        return pts_all, conf_all

    xywh_cat = torch.cat(xywh_list, dim=0)
    conf_cat = torch.cat(conf_list, dim=0)

    xywh_np = xywh_cat.cpu().numpy()
    conf_np = conf_cat.cpu().numpy()

    idx = 0
    for n in range(N):
        for m in range(M):
            num = counts[n*M + m]
            if num == 0:
                continue
            xywh = xywh_np[idx:idx+num]
            conf = conf_np[idx:idx+num]
            idx += num

            mask = conf >= min_conf
            xywh = xywh[mask]
            conf = conf[mask]

            num_keep = min(len(xywh), O)
            if num_keep > 0:
                pts_all[n, m, :num_keep, :] = xywh[:num_keep]
                conf_all[n, m, :num_keep] = conf[:num_keep]

    return pts_all, conf_all



# ========================================================================
# ========================================================================


def estiamte3D_work(DATA,configuration,P_all, K_all, dist_all, C_all, pair_transforms, work = True, conf_bat = None):

    selection_3D = []
    analyze_3D = {}
    info_time = {}

    N = configuration["N"]
    M = configuration["M"]
    O = configuration["O"]
    min_conf = configuration["min_conf"]
    bounds = configuration["bounds"]
    min_angle_deg = configuration["min_angle_deg"]
    max_err = configuration["max_err"]

    t1 = time.perf_counter()
    if work:
        pts_bat, conf_bat = extract_results_to_matrices_batch(DATA, N, M, O, min_conf)
    else:
        pts_bat = DATA

    t2 = time.perf_counter()
    pts_undist = undistort_points_numpy_iterative(pts_bat, K_all, dist_all, max_iter=7)

    t3 = time.perf_counter()
    all_3D, meta = triangulate_all_variants(pts_undist, P_all)

    t4 = time.perf_counter()
    analyze_3D = analyze_triangulated_points_B(all_3D, conf_bat, meta, pts_undist, P_all, C_all,
                                             min_angle_deg=min_angle_deg,
                                             bounds=bounds,
                                             max_err = max_err)
    t5 = time.perf_counter()
    selection_3D = process3D_pairCams(analyze_3D, pair_transforms, labels_len = N)
    t6 = time.perf_counter()

    info_time = {
        "GPU --> CPU": t2 - t1,
        "undistort": 1000 * (t3 - t2),
        "get3D": 1000 * (t4 - t3),
        "analyze_3D": 1000 * (t5 - t4),
        "selection_3D": 1000 * (t6 - t5),
    }

    return selection_3D, analyze_3D, info_time






# INNE ==========================================================================
# ===============================================================================
# ===============================================================================



def compute_all_3D_points(entries, transforms, apply_similarity):
    """
    Przelicza punkty 3D dla wszystkich wpisów w 'entries'
    z użyciem odpowiedniego similarity transform (sim)
    na podstawie pary kamer ('cameras').

    Zwraca listę 'list_3D_all' z przeliczonymi punktami.
    """
    list_3D_all = []

    for e in entries:
        point = np.array(e["point"], dtype=float)
        pair = tuple(sorted(e["cameras"]))

        # znajdź transformację (jeśli istnieje)
        key = str(pair)
        sim = transforms.get(key, {}).get("similarity", None)

        # przelicz punkt jeśli transformacja istnieje
        if sim is not None:
            point_transformed = apply_similarity(point, sim)
        else:
            point_transformed = point  # jeśli brak transformacji, zostaw oryginał

        list_3D_all.append(point_transformed.tolist())

    return list_3D_all


def find_cluster_close_points(points_3D, max_distance=0.2):
    """
    Szuka grupy punktów 3D, które leżą blisko siebie (<= max_distance między sobą).
    Zwraca:
      - listę punktów, które spełniają warunek,
      - średnią (np.array) z tych punktów,
      - indeksy tych punktów w oryginalnej liście.

    Jeśli nie znaleziono grupy: ([], None, [])
    """
    if not points_3D:
        return [], None, []

    pts = np.array(points_3D, dtype=float)
    n = len(pts)

    # --- oblicz macierz odległości między wszystkimi punktami ---
    distances = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)

    best_group = []
    for i in range(n):
        # znajdź punkty bliższe niż max_distance do punktu i
        close_idx = np.where(distances[i] <= max_distance)[0]
        if len(close_idx) > len(best_group):
            best_group = close_idx

    if len(best_group) > 1:
        cluster_points = pts[best_group]
        avg_point = np.mean(cluster_points, axis=0)
        return cluster_points.tolist(), avg_point.tolist(), best_group.tolist()
    else:
        return [], None, []


# ====================================================

def process3D_pairCams(meta_data3D, pair_transforms, labels_len):
    """
    Zwraca list_3D o długości = labels_len.
      • preferuje parę odniesienia (lead_pair z transformów lub domyślnie ('cL','cR')),
      • jeśli brak danych z tej pary → globalne min_err,
      • mapowanie similarity (jeśli dostępne),
      • brak danych → [].
    """

    # --- Ustal parę odniesienia i transformacje ---
    if pair_transforms and pair_transforms.get("lead_pair"):
        lead_pair = tuple(pair_transforms["lead_pair"])
        transforms = pair_transforms.get("pairs", {})
        # print(f"[INFO] Wykorzystano parę odniesienia z transformów: {lead_pair}")
    else:
        # lead_pair = ("cL", "cR")  # domyślna para odniesienia
        lead_pair = (0, 1)  # domyślna para odniesienia
        transforms = {}
        # print("[INFO] pair_transforms puste → użyto domyślnej pary ('cL','cR')")

    # --- Grupowanie danych po numerze chwili ---
    per_frame = {}
    for e in meta_data3D:
        n = int(e["n"])
        per_frame.setdefault(n, []).append(e)

    list_3D = []

    for n in range(1, labels_len + 1):
        # print(f"\nn: {n}")
        entries = per_frame.get(n, [])
        # print(f"entries: {entries}")
        if not entries:
            list_3D.append([])
            continue

        best_point = None
        best_err = np.inf
        used_pair = None
        mapped_needed = False

        # --- 1️⃣ Preferowana para odniesienia (lead_pair lub ('cL','cR')) ---
        for e in entries:
            pair = tuple(sorted(e["cameras"]))
            if pair == lead_pair and is_valid_entry(e):
                err = e["min_err"]
                if err < best_err:
                    best_err = err
                    best_point = np.array(e["point"], dtype=float)
                    used_pair = pair
                    mapped_needed = False

        # print(f"Etap 1: best_err: {best_err:5.1f}; used_pair: {used_pair}; best_point: {best_point}")

        # --- 2️⃣ Jeśli brak, szukamy globalnego minimum błędu ---
        if best_point is None:
            for e in entries:
                if not is_valid_entry(e):
                    continue
                pair = tuple(sorted(e["cameras"]))
                err = e["min_err"]
                if err < best_err:
                    best_err = err
                    best_point = np.array(e["point"], dtype=float)
                    used_pair = pair
                    mapped_needed = (lead_pair and pair != lead_pair)

            # print(f"Etap 2: best_err: {best_err:5.1f}; used_pair: {used_pair}; best_point: {best_point}")

        # --- 3️⃣ Mapowanie do układu odniesienia (jeśli potrzeba) ---
        if (best_point is not None): #and (best_err<=10):
            if mapped_needed and used_pair is not None and transforms:
                key = str(tuple(sorted(used_pair)))
                sim = transforms.get(key, {}).get("similarity", None)
                best_point = apply_similarity(best_point, sim)
            list_3D.append(best_point.tolist())
            # if best_err<10:
            #     list_3D.append(best_point.tolist())
            # else:
            #     list_3D.append([])
        else:
            list_3D.append([])

        # print(f"Etap end: best_err: {best_err:5.1f}; used_pair: {used_pair}; best_point: {best_point}")

        # list_3D_all = compute_all_3D_points(entries, transforms, apply_similarity)
        #
        # print("list_3D_all:")
        # for p in list_3D_all:
        #     print(p)

        #
        # cluster_points, avg_point, idx = find_cluster_close_points(list_3D_all, max_distance=0.2)
        # print(f"--\n{cluster_points}\n{avg_point}")
        #
        # if avg_point is not None:
        #     list_3D[-1]=avg_point


    return list_3D


def apply_similarity(x, sim_dict):
    """Zastosuj similarity transform: y = s * R * x + t."""
    if not sim_dict or sim_dict.get("type") in (None, "insufficient"):
        return x
    s = sim_dict.get("s", 1.0)
    R = np.array(sim_dict.get("R", np.eye(3)))
    t = np.array(sim_dict.get("t", [0.0, 0.0, 0.0]))
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return s * (R @ x) + t
    return (s * (R @ x.T)).T + t


def is_valid_entry(e):
    """Sprawdź, czy wpis 3D jest poprawny."""
    return (
        e.get("check_cheirality", False)
        and e.get("in_bounds", False)
        and np.isfinite(e.get("min_err", np.inf))
        and np.isfinite(e.get("min_angle_deg", 0.0))
    )


# =================================================================

def load_calibration(calib_path):
    """Wczytuje P_all, K_all, dist_all z pliku projMtx_all.json"""
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cams = data["P_undistort"]["cameras"]
    P_all, K_all, dist_all = [], [], []
    for cam in CAMERAS:
        cam_data = cams[cam]
        P_all.append(np.array(cam_data["P"], dtype=np.float32))
        K_all.append(np.array(cam_data["K"], dtype=np.float32))
        dist_all.append(np.array(cam_data["k1"], dtype=np.float32)[0])
    P = np.stack(P_all)
    K = np.stack(K_all)
    dist = np.array(dist_all)
    C = precompute_camera_centers(P)
    # return np.stack(P_all), np.stack(K_all), np.array(dist_all)
    return P, K, dist, C


def precompute_camera_centers(P_all):
    """Oblicz centra kamer (C = -R^T t)."""
    M = P_all.shape[0]
    C_all = np.zeros((M, 3))
    for m in range(M):
        P = P_all[m]
        R = P[:, :3]
        t = P[:, 3]
        C_all[m] = -np.linalg.inv(R) @ t
    return C_all


def load_cam_trensforms(transform_path = None):
    if transform_path is not None:
        with open(transform_path, "r", encoding="utf-8") as f:
            pair_transforms = json.load(f)
    else:
        pair_transforms = {}
    return pair_transforms