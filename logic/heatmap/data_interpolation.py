import math
import numpy as np
import pandas as pd

from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import Point, MultiPoint

import logging

logger = logging.getLogger(__name__)

# # --- Funzioni di interpolazione dei dati ---

# Costruisce una griglia quadrata di punti (celle) dentro al bounding box.
def build_square_grid(xmin, xmax, ymin, ymax, max_cells=100000, N=8):
    """
    Costruisce una griglia quadrata di punti (celle) dentro al bounding box.
    
    Parametri
    ---------
    xmin, xmax, ymin, ymax : float
        Limiti del bounding box (in metri, coordinate UTM).
    max_cells : int, opzionale
        Numero massimo di celle consentito (default 100k).
    default_step : float, opzionale
        Passo iniziale in metri (default 100).
    N : denominatore per l'incremento per iterazione (default 8)
        (es: N = 8 c'è un aumento di 1/8 dello step ad ogni iterazione)
    
    Ritorna
    -------
    grid_x, grid_y : 2D array
        Coordinate della griglia.
    step : int
        Passo effettivo usato.
    nx, ny : int
        Numero celle in x e y.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    
    logger.info(f"Bounding Box size: dx={dx:.1f} m, dy={dy:.1f} m")

    # Step iniziale
    step = 10

    # Calcola quanti punti genererebbe
    nx = int(math.ceil(dx / step))
    ny = int(math.ceil(dy / step))
    total = nx * ny

    # Se il numero di celle supera max_cells, aumenta step fino a stare sotto
    while total > max_cells:
        step *= (1 + 1/N)  # aumento progressivo
        step = int(step)
        nx = int(math.ceil(dx / step))
        ny = int(math.ceil(dy / step))
        total = nx * ny

    # Costruisci la griglia
    x_values = np.linspace(xmin, xmax, nx)
    y_values = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    return grid_x, grid_y, step, nx, ny

# Interpolazione principale
def interpolate(xy_known, values, grid_x, grid_y, model, **kwargs):
    if model == "idw":
        method = idw_interpolation
    elif model == "akima":
        method = akima_interpolation
        
    return method(xy_known, values, grid_x, grid_y, **kwargs)

# Funzione di interpolazione IDW
def idw_interpolation(xy_known, values, grid_x, grid_y, **kwargs):
    """
    Esegue IDW su una griglia.
    
    xy_known : array (N,2) con le coordinate note (X,Y)
    values   : array (N,) con i valori nei punti noti
    grid_x, grid_y : meshgrid numpy con le coordinate della griglia
    **kwargs : parametri aggiuntivi (power)
    
    Ritorna grid_z con le stime IDW
    """
    
    power = kwargs.get('power', 2)
    
    # Reshape per broadcasting
    gx = grid_x.flatten()
    gy = grid_y.flatten()
    points = np.vstack((gx, gy)).T  # (M,2)

    # Calcola distanze da ogni punto della griglia a ogni sensore
    dist = np.sqrt(((points[:, None, :] - xy_known[None, :, :]) ** 2).sum(axis=2))  # (M,N)

    # Evita divisione per zero (se un punto coincide con un sensore)
    dist[dist == 0] = 1e-10

    weights = 1 / dist**power  # (M,N)
    weights /= weights.sum(axis=1)[:, None]  # normalizza

    # Stima i valori
    z = np.sum(weights * values[None, :], axis=1)

    return z.reshape(grid_x.shape)

# Funzione di interpolazione Akima 2D (semplificata)
def akima_interpolation(xy_known, val_known, grid_x, grid_y, **kwargs):
    """
    Akima interpolator inside convex hull + smooth 1-cell expansion.
    """

    cell_size = kwargs.get("cell_size", None)
    eps = 1e-6
    
    df = pd.DataFrame({
    "x": xy_known[:, 0],
    "y": xy_known[:, 1],
    "value": val_known
    })

    # raggruppa per coordinate e media dei valori
    df_agg = df.groupby(["x","y"], as_index=False).mean()

    xy_known = df_agg[["x","y"]].values
    val_known = df_agg["value"].values

    # ------------------------------------------------------------
    # 1) Convex hull originale
    # ------------------------------------------------------------
    hull_original = MultiPoint(xy_known).convex_hull

    # ------------------------------------------------------------
    # 2) Akima inside hull
    # ------------------------------------------------------------
    tri = Delaunay(xy_known)
    interp = LinearNDInterpolator(tri, val_known)

    gx = grid_x.ravel()
    gy = grid_y.ravel()
    pts = np.column_stack((gx, gy))

    grid_z = np.full_like(grid_x, np.nan, dtype=float)

    inside_original = np.array([hull_original.contains(Point(p)) for p in pts])
    inside_idx = np.where(inside_original)[0]

    if inside_idx.size > 0:
        vals = interp(pts[inside_idx])
        ok = np.isfinite(vals)
        grid_z.ravel()[inside_idx[ok]] = vals[ok]

    # ------------------------------------------------------------
    # 3) Hull espansa di una cella
    # ------------------------------------------------------------
    expand_dist = cell_size * 1
    hull_expanded = hull_original.buffer(expand_dist)

    inside_expanded = np.array([hull_expanded.contains(Point(p)) for p in pts])
    border_idx = np.where(inside_expanded & (~inside_original))[0]

    if border_idx.size == 0:
        return grid_z

    P = pts[border_idx]

    # ------------------------------------------------------------
    # 4) Per ogni punto esterno: media pesata dei 4 vicini interni più vicini
    # ------------------------------------------------------------
    inside_pts = pts[inside_idx]
    inside_vals = grid_z.ravel()[inside_idx]

    tree_inside = cKDTree(inside_pts)

    # prendiamo 4 vicini
    d, nn = tree_inside.query(P, k=4)

    d = d + eps  # evita divisione per zero
    w = 1.0 / d  # pesi ~ nearest ma morbidi

    blended = np.sum(w * inside_vals[nn], axis=1) / np.sum(w, axis=1)

    # assegna valori alla fascia esterna
    grid_z.ravel()[border_idx] = blended

    return grid_z