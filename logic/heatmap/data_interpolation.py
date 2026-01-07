''' Snap4city Computing HEATMAP - Data Interpolation Module.
    Copyright (C) 2024 DISIT Lab http://www.disit.org - University of Florence
'''

import math
import numpy as np
import logging
from abc import ABC, abstractmethod
from shapely.geometry import MultiPoint
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Akima1DInterpolator
from shapely import contains_xy

logger = logging.getLogger(__name__)

class Interpolator(ABC):
    """
    Abstract Base Class for spatial interpolation.
    
    Handles grid generation and provides a centralized logic to ensure the 
    total number of grid cells stays within defined memory and platform limits.
    """
    def __init__(self, xy_known, val_known, max_cells=10000, base_cell_size=10.0, bbox=None):
        """
        Initializes the Interpolator.

        Args:
            xy_known (np.ndarray): Array of known sensor coordinates (UTM).
            val_known (np.ndarray): Array of sensor values.
            max_cells (int): Maximum allowed number of cells in the resulting grid.
            base_cell_size (float): Initial target size for each cell (meters).
            bbox (shapely.geometry.Polygon): Bounding Box of the area of interest.
        """
        self.xy_known = xy_known
        self.val_known = val_known
        self.max_cells = max_cells
        self.cell_size = base_cell_size
        self.bbox = bbox
        self.step_increment = 12.5  

    @staticmethod
    def build(method, xy_known, val_known, max_cells, bbox):
        """
        Factory method to instantiate the correct Interpolator subclass.
        """
        if method == 'idw':
            return IDWInterpolator(xy_known, val_known, max_cells, bbox=bbox)
        elif method == 'akima':
            return AkimaInterpolator(xy_known, val_known, max_cells, bbox=bbox)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def run(self):
        """
        Executes the interpolation workflow: grid generation followed by value calculation.
        """
        grid_x, grid_y = self.build_grid()
        grid_z = self.interpolate(grid_x, grid_y)
        return grid_x, grid_y, grid_z, self.cell_size

    def refine_step_to_limit(self, start_step):
        """
        Incrementally increases the step size until the total number of cells 
        is within the max_cells limit for the given BBox.
        """
        xmin, ymin, xmax, ymax = self.bbox.bounds
        dx, dy = xmax - xmin, ymax - ymin
        
        step = max(1.0, start_step)
        nx, ny = int(math.ceil(dx / step)), int(math.ceil(dy / step))
        total = nx * ny

        while total > self.max_cells:
            step *= 1 + self.step_increment / 100
            nx, ny = int(math.ceil(dx / step)), int(math.ceil(dy / step))
            total = nx * ny
            
        return math.ceil(step)

    @abstractmethod
    def build_grid(self):
        """Generates the coordinate meshgrid."""
        pass

    @abstractmethod
    def interpolate(self, grid_x, grid_y):
        """Calculates interpolated values for the meshgrid."""
        pass

class IDWInterpolator(Interpolator):
    """
    Inverse Distance Weighting (IDW) Interpolator with Steep Distance-Based Decay.
    
    Values are calculated as a weighted average of known points. An aggressive 
    exponential fading factor is applied to cells beyond a certain distance from 
    sensors to eliminate artifacts in unmonitored areas.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.power = 4
        self.max_dist = 50.0   # Zone of full influence (meters)
        self.steepness = 5.0  # Fading rate (lower = more aggressive cut)

    def build_grid(self):
        """Generates a grid ensuring total cells <= max_cells."""
        self.cell_size = self.refine_step_to_limit(self.cell_size)
        xmin, ymin, xmax, ymax = self.bbox.bounds
        x_edges = np.arange(xmin, xmax + self.cell_size, self.cell_size)
        y_edges = np.arange(ymin, ymax + self.cell_size, self.cell_size)
        return np.meshgrid(x_edges, y_edges)

    def interpolate(self, grid_x, grid_y):
        """
        Performs IDW interpolation with an aggressive exponential decay.
        """
        gx, gy = grid_x.flatten(), grid_y.flatten()
        points = np.vstack((gx, gy)).T
        
        # Calculate distances to all sensors
        dist = np.sqrt(((points[:, None, :] - self.xy_known[None, :, :]) ** 2).sum(axis=2))
        
        # Determine distance to the nearest sensor for each cell
        min_dist = np.min(dist, axis=1)
        
        dist[dist == 0] = 1e-10
        weights = 1 / dist**self.power
        weights /= weights.sum(axis=1)[:, None]
        
        # Base IDW calculation
        z = np.sum(weights * self.val_known[None, :], axis=1)
        
        # Aggressive decay: 1.0 within max_dist, then exponential drop-off
        fade_factor = np.where(
            min_dist <= self.max_dist, 
            1.0, 
            np.exp(-(min_dist - self.max_dist) / self.steepness)
        )
        
        z *= fade_factor
        return z.reshape(grid_x.shape)

class AkimaInterpolator(Interpolator):
    """
    Advanced Akima 2D Interpolator using row-column passes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hull_expand_cells = 3  
        points = MultiPoint(self.xy_known)
        self.hull_orig = points.convex_hull
        self.eps = 1e-12

    def build_grid(self):
        """Generates a high-resolution grid scaled to the sensor area."""
        xmin, ymin, xmax, ymax = self.bbox.bounds
        area_bbox = (xmax - xmin) * (ymax - ymin)
        area_hull = self.hull_orig.area
        ratio = area_hull / area_bbox
        
        desired_step = self.cell_size
        if 0 < ratio < 1.0:
            desired_step *= math.sqrt(ratio)
            
        self.cell_size = self.refine_step_to_limit(desired_step)
        
        logger.debug(f"Akima Grid - Ratio: {ratio:.4f}, Final Step: {self.cell_size}")
        
        x_edges = np.arange(xmin, xmax + self.cell_size, self.cell_size)
        y_edges = np.arange(ymin, ymax + self.cell_size, self.cell_size)
        return np.meshgrid(x_edges, y_edges)

    def interpolate(self, grid_x, grid_y):
        """Performs 2D Akima interpolation via dual-pass logic."""
        hull_expanded = self.hull_orig.buffer(self.cell_size * self.hull_expand_cells)
        mask_ext = contains_xy(hull_expanded, grid_x, grid_y)
        
        tree = cKDTree(self.xy_known)
        grid_z_rows = np.full(grid_x.shape, np.nan)
        grid_z_cols = np.full(grid_x.shape, np.nan)
        
        power = 3.5
        rows, cols = np.where(mask_ext)
        if rows.size == 0: 
            return np.full(grid_x.shape, np.nan)

        # --- ROW-WISE PASS ---
        for r in range(rows.min(), rows.max() + 1):
            idx_in_hull = np.where(mask_ext[r, :])[0]
            if len(idx_in_hull) < 2: continue

            x_start, x_end = grid_x[r, idx_in_hull[0]], grid_x[r, idx_in_hull[-1]]
            x_control = np.linspace(x_start, x_end, 5)
            y_const = grid_y[r, 0]
            
            pts_c = np.column_stack((x_control, np.full(5, y_const)))
            dists, idxs = tree.query(pts_c, k=min(6, len(self.xy_known)))
            
            z_control = []
            for j in range(5):
                d = dists[j] + self.eps
                w = 1.0 / (d**power)
                z_control.append(np.sum(w * self.val_known[idxs[j]]) / np.sum(w))
            
            try:
                ak = Akima1DInterpolator(x_control, z_control)
                grid_z_rows[r, idx_in_hull] = ak(grid_x[r, idx_in_hull])
            except:
                grid_z_rows[r, idx_in_hull] = np.interp(grid_x[r, idx_in_hull], x_control, z_control)

        # --- COLUMN-WISE PASS ---
        for c in range(cols.min(), cols.max() + 1):
            idx_in_hull = np.where(mask_ext[:, c])[0]
            if len(idx_in_hull) < 2: continue

            y_start, y_end = grid_y[idx_in_hull[0], c], grid_y[idx_in_hull[-1], c]
            y_control = np.linspace(y_start, y_end, 5)
            x_const = grid_x[0, c]
            
            pts_c = np.column_stack((np.full(5, x_const), y_control))
            dists, idxs = tree.query(pts_c, k=min(6, len(self.xy_known)))
            
            z_control = []
            for j in range(5):
                d = dists[j] + self.eps
                w = 1.0 / (d**power)
                z_control.append(np.sum(w * self.val_known[idxs[j]]) / np.sum(w))
            
            try:
                ak = Akima1DInterpolator(y_control, z_control)
                grid_z_cols[idx_in_hull, c] = ak(grid_y[idx_in_hull, c])
            except:
                grid_z_cols[idx_in_hull, c] = np.interp(grid_y[idx_in_hull, c], y_control, z_control)

        with np.errstate(all='ignore'):
            grid_z = np.nanmean([grid_z_rows, grid_z_cols], axis=0)

        for i in range(len(self.val_known)):
            ix = np.abs(grid_x[0, :] - self.xy_known[i, 0]).argmin()
            iy = np.abs(grid_y[:, 0] - self.xy_known[i, 1]).argmin()
            grid_z[max(0,iy-1):iy+2, max(0,ix-1):ix+2] = self.val_known[i]

        grid_z[~mask_ext] = np.nan
        nan_mask = np.isnan(grid_z)
        grid_z_filled = np.where(nan_mask, np.nanmean(grid_z), grid_z)
        smoothed_z = gaussian_filter(grid_z_filled, sigma=1.2)
        smoothed_z[nan_mask] = np.nan
        
        return smoothed_z