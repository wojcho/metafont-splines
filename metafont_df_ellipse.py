import numpy as np
from scipy.spatial import cKDTree
import math
from skimage import measure
from scipy.spatial.transform import Rotation
from typing import Optional

from fit_beziers import bezier_cubic

def velocity(st: float, ct: float, sf: float, cf: float, t: float) -> float:
    """Compute velocity ratio for control points."""
    acc = (st - sf / 16.0) * (ct - cf)
    num = 2.0 + acc * math.sqrt(2)
    denom = 3.0 + ct * (497706707 / 2**28) + cf * (307599661 / 2**28)
    if t != 1.0:
        num = num / t
    if num / 4.0 >= denom:
        result = 4.0
    else:
        result = num / denom
    return result

def n_arg(x: float, y: float) -> float:
    """Calculate angle of vector (x,y) in degrees."""
    if x == 0 and y == 0:
        return 0
    return math.degrees(math.atan2(y, x))

def make_choices(
    knots: list[tuple[float, float]],
    tensions: Optional[list[tuple[float, float]]] = None,
    curls: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    directions: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
) -> list[tuple[float, float]]:
    """Calculate control points for Hobby's spline."""
    n = len(knots) - 1
    if n < 1:
        return []

    # Initialize arrays
    delta_x = [0] * (n + 1)
    delta_y = [0] * (n + 1)
    delta = [0] * (n + 1)
    psi = [0] * (n + 1)
    theta = [0] * (n + 1)
    uu = [0] * (n + 1)
    vv = [0] * (n + 1)
    ww = [0] * (n + 1)

    # Default tensions and curls
    if tensions is None:
        tensions = [(1.0, 1.0)] * n
    if curls is None:
        curls = [(None, None)] * (n + 1)
        curls[0] = (1.0, 1.0)
        curls[-1] = (1.0, 1.0)
    if directions is None:
        directions = [(None, None)] * (n + 1)

    # Calculate deltas and turning angles
    for k in range(n):
        delta_x[k] = knots[k + 1][0] - knots[k][0]
        delta_y[k] = knots[k + 1][1] - knots[k][1]
        delta[k] = math.sqrt(delta_x[k]**2 + delta_y[k]**2)
        if k > 0:
            sine = delta_y[k-1] / delta[k-1] if delta[k-1] != 0 else 0
            cosine = delta_x[k-1] / delta[k-1] if delta[k-1] != 0 else 1
            dx = delta_x[k] * cosine + delta_y[k] * sine
            dy = delta_y[k] * cosine - delta_x[k] * sine
            psi[k] = n_arg(dx, dy)
    psi[n] = 0

    # Handle simple case: n=1 with curls or directions
    if n == 1:
        if directions[0][1] is not None and directions[1][0] is not None:
            aa = n_arg(delta_x[0], delta_y[0])
            theta[0] = directions[0][1] - aa
            phi = -(directions[1][0] - aa)
            return prepare_controls(knots, 0, theta[0], phi, tensions[0])
        elif curls[0][1] is not None and curls[1][0] is not None:
            control_points = []
            rt = abs(tensions[0][1])
            lt = abs(tensions[0][0])
            aa = n_arg(delta_x[0], delta_y[0])
            st = math.sin(math.radians(aa))
            ct = math.cos(math.radians(aa))
            sf = st
            cf = ct
            rr = velocity(st, ct, sf, cf, rt)
            ss = velocity(sf, cf, st, ct, lt)
            right_x = knots[0][0] + delta_x[0] * rr
            right_y = knots[0][1] + delta_y[0] * rr
            left_x = knots[1][0] - delta_x[0] * ss
            left_y = knots[1][1] - delta_y[0] * ss
            control_points.append((right_x, right_y))
            control_points.append((left_x, left_y))
            return control_points

    # Set up equations
    if directions[0][1] is not None:
        vv[0] = directions[0][1] - n_arg(delta_x[0], delta_y[0])
        if abs(vv[0]) > 180:
            vv[0] -= 360 if vv[0] > 0 else -360
        uu[0] = 0
        ww[0] = 0
    elif curls[0][1] is not None:
        cc = curls[0][1]
        lt = abs(tensions[0][0])
        rt = abs(tensions[0][1])
        uu[0] = curl_ratio(cc, rt, lt)
        vv[0] = -psi[1] * uu[0]
        ww[0] = 0
    else:
        uu[0] = 0
        vv[0] = 0
        ww[0] = 1.0

    # Solve equations
    for k in range(1, n):
        rt = abs(tensions[k-1][1])
        lt = abs(tensions[k][0])
        if rt == 1.0:
            aa = 0.5
            dd = 2 * delta[k]
        else:
            aa = 1.0 / (3 * rt - 1)
            dd = delta[k] * (3 - 1.0 / rt)
        if lt == 1.0:
            bb = 0.5
            ee = 2 * delta[k-1]
        else:
            bb = 1.0 / (3 * lt - 1)
            ee = delta[k-1] * (3 - 1.0 / lt)
        cc = 1.0 - uu[k-1] * aa
        dd = dd * cc
        if lt != rt:
            if lt < rt:
                ff = lt / rt
                ff = ff * ff
                dd = dd * ff
            else:
                ff = rt / lt
                ff = ff * ff
                ee = ee * ff
        ff = ee / (ee + dd) if (ee + dd) != 0 else 0.5
        uu[k] = ff * bb
        acc = -psi[k+1] * uu[k]
        if curls[k-1][1] is not None:
            ww[k] = 0
            vv[k] = acc - psi[1] * (1 - ff)
        else:
            ff = (1 - ff) / cc
            acc = acc - psi[k] * ff
            ff = ff * aa
            vv[k] = acc - vv[k-1] * ff
            ww[k] = -ww[k-1] * ff if ww[k-1] != 0 else 0

    # Handle final equation
    if directions[n][0] is not None:
        theta[n] = directions[n][0] - n_arg(delta_x[n-1], delta_y[n-1])
        if abs(theta[n]) > 180:
            theta[n] -= 360 if theta[n] > 0 else -360
    elif curls[n][0] is not None:
        cc = curls[n][0]
        lt = abs(tensions[n-1][0])
        rt = abs(tensions[n-1][1])
        ff = curl_ratio(cc, lt, rt)
        theta[n] = -(vv[n-1] * ff) / (1 - ff * uu[n-1]) if (1 - ff * uu[n-1]) != 0 else 0
    else:
        aa = 0
        bb = 1.0
        k = n
        while True:
            k -= 1
            if k == 0:
                k = n
            aa = vv[k] - aa * uu[k]
            bb = ww[k] - bb * uu[k]
            if k == n:
                break
        aa = aa / (1 - bb) if (1 - bb) != 0 else 0
        theta[n] = aa
        vv[0] = aa
        for k in range(1, n):
            vv[k] = vv[k] + aa * ww[k]

    # Back-substitute to find theta values
    for k in range(n-1, -1, -1):
        theta[k] = vv[k] - theta[k+1] * uu[k]

    # Assign control points
    control_points = []
    for k in range(n):
        st = math.sin(math.radians(theta[k]))
        ct = math.cos(math.radians(theta[k]))
        phi = -psi[k+1] - theta[k+1]
        sf = math.sin(math.radians(phi))
        cf = math.cos(math.radians(phi))
        rt = abs(tensions[k][1])
        lt = abs(tensions[k][0])
        rr = velocity(st, ct, sf, cf, rt)
        ss = velocity(sf, cf, st, ct, lt)
        
        # Bounding triangle check
        if (st >= 0 and sf >= 0) or (st <= 0 and sf <= 0):
            sine = abs(st) * cf + abs(sf) * ct
            if sine > 0:
                sine = sine * 0.99
                if tensions[k][1] < 0:
                    if abs(sf) < rr * sine:
                        rr = abs(sf) / sine
                if tensions[k][0] < 0:
                    if abs(st) < ss * sine:
                        ss = abs(st) / sine
        
        right_x = knots[k][0] + (delta_x[k] * ct - delta_y[k] * st) * rr
        right_y = knots[k][1] + (delta_y[k] * ct + delta_x[k] * st) * rr
        left_x = knots[k+1][0] - (delta_x[k] * cf + delta_y[k] * sf) * ss
        left_y = knots[k+1][1] - (delta_y[k] * cf - delta_x[k] * sf) * ss
        control_points.append((right_x, right_y))
        control_points.append((left_x, left_y))
    return control_points

def curl_ratio(gamma: float, a_tension: float, b_tension: float) -> float:
    """Calculate curl ratio for endpoint equations."""
    alpha = 1.0 / a_tension
    beta = 1.0 / b_tension
    if alpha <= beta:
        ff = alpha / beta
        ff = ff * ff
        gamma = gamma * ff
        beta = beta
        denom = gamma * alpha + (3 - beta)
        num = gamma * (3 - alpha) + beta
    else:
        ff = beta / alpha
        ff = ff * ff
        beta = beta * ff
        denom = gamma * alpha + (3 - beta)
        num = gamma * (3 - alpha) + beta
    return min(num / denom, 4.0) if denom != 0 else 4.0

def prepare_controls(
    knots: list[tuple[float, float]],
    k: int,
    theta: float,
    phi: float,
    tension: tuple[float, float],
) -> list[tuple[float, float]]:
    """Set control points for a single segment."""
    st = math.sin(math.radians(theta))
    ct = math.cos(math.radians(theta))
    sf = math.sin(math.radians(phi))
    cf = math.cos(math.radians(phi))
    rt = abs(tension[1])
    lt = abs(tension[0])
    rr = velocity(st, ct, sf, cf, rt)
    ss = velocity(sf, cf, st, ct, lt)
    
    right_x = knots[k][0] + (delta_x[k] * ct - delta_y[k] * st) * rr
    right_y = knots[k][1] + (delta_y[k] * ct + delta_x[k] * st) * rr
    left_x = knots[k+1][0] - (delta_x[k] * cf + delta_y[k] * sf) * ss
    left_y = knots[k+1][1] - (delta_y[k] * cf - delta_x[k] * sf) * ss
    return [(right_x, right_y), (left_x, left_y)]

def sample_curve_all_segments(
    knots: list[tuple[float, float]],
    control_points: list[tuple[float, float]],
    samples_per_seg: int = 200,
) -> np.ndarray:
    pts = []
    m = len(knots) - 1
    t = np.linspace(0.0, 1.0, samples_per_seg)
    for i in range(m):
        P0 = np.asarray(knots[i], dtype=float)
        P1 = np.asarray(control_points[2*i], dtype=float)
        P2 = np.asarray(control_points[2*i+1], dtype=float)
        P3 = np.asarray(knots[i+1], dtype=float)
        seg = bezier_cubic(P0, P1, P2, P3, t)
        pts.extend(seg)
    return np.array(pts)

def distance_field_from_samples(
    samples: np.ndarray,
    query_points: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    tree = cKDTree(samples)
    dists, _ = tree.query(query_points, k=1)
    return dists.reshape(shape)

def extract_iso_contours(
    XX: np.ndarray,
    YY: np.ndarray,
    field: np.ndarray,
    level: float,
) -> list[np.ndarray]:
    """
    Return list of contours as arrays of (x,y) coordinates for iso-distance=level.
    Uses pixel coordinates -> map to XY coordinates using XX, YY mesh.
    """
    # skimage expects image with origin at top-left; our XX/YY follow matplotlib (origin="lower")
    # so pass field as-is but remember rows correspond to y-grid.
    contours = measure.find_contours(field, level=level)
    result = []
    x0, x1 = XX[0,0], XX[0,-1]
    y0, y1 = YY[0,0], YY[-1,0]
    ny, nx = field.shape
    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)
    for c in contours:
        # c is Nx2 array of (row, col) coordinates in image space
        rows = c[:, 0]
        cols = c[:, 1]
        xs = x0 + cols * dx
        ys = y0 + rows * dy
        pts = np.column_stack([xs, ys])
        result.append(pts)
    return result

def affine_transform_points(
    points: list[tuple[float, float]] | np.ndarray,
    matrix: np.ndarray,
    translation: np.ndarray = np.zeros(2),
) -> np.ndarray:
    """Apply 2D affine transform: p -> M p + t"""
    pts = np.array(points)
    return (pts @ matrix.T) + translation

def make_ellipse_transform(
    angle_deg: float,
    semi_major: float,
    semi_minor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns T and T_inv (2x2 matrices)"""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]]) # rotation
    S = np.diag([semi_major, semi_minor]) # scaling
    T = R @ S # ellipse = T * unit_circle
    T_inv = np.linalg.inv(T)
    return T, T_inv

def bounds_from_knots(
    knots: list[tuple[float, float]],
    pad: float = 1.0,
) -> tuple[float, float, float, float]:
    pts = np.asarray(knots)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return xmin - pad, ymin - pad, xmax + pad, ymax + pad

def find_df_and_contours(
    knots: list[tuple[float, float]],
    ellipse_angle_deg: float = 45,
    ellipse_semi_major: float = 0.15,
    ellipse_semi_minor: float = 0.05,
    grid_x_samples: int = 300,
    grid_y_samples: int = 200,
    grid_padding: float = 1.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[float],
    list[tuple[float, float]],
    np.ndarray,
    list[np.ndarray],
]:
    # Prepare distance field for known knots, and find contours
    # Make a curve, inverse transform, find distance field and contours, forward transform
    T, T_inv = make_ellipse_transform(angle_deg=ellipse_angle_deg, semi_major=ellipse_semi_major, semi_minor=ellipse_semi_minor) # Transform matrices
    xmin, ymin, xmax, ymax = bounds_from_knots(knots, pad=grid_padding)
    x_grid = np.linspace(xmin, xmax, grid_x_samples)
    y_grid = np.linspace(ymin, ymax, grid_y_samples)
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="xy")
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])
    grid_points_t = affine_transform_points(grid_points, T_inv)
    XX_t = grid_points_t[:,0].reshape(XX.shape)
    YY_t = grid_points_t[:,1].reshape(YY.shape)
    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]

    knots_t = affine_transform_points(knots, T_inv)
    control_points = make_choices(knots)
    control_points_t = affine_transform_points(control_points, T_inv)
    samples_t = sample_curve_all_segments(
        knots_t,
        control_points_t,
        samples_per_seg=300
    )

    field_t = distance_field_from_samples(
        samples_t,
        grid_points_t,
        XX.shape
    )
    # Obtain Minkowski sum of an ellipse and Hobby spline curves
    # Without transforms it would be Minkowski sum of a circle and Hobby spline curves
    contours = extract_iso_contours(XX, YY, field_t, level=1.0) # It is a trick, field is in different transformed space, with transformed different distance, but coordinates are in normal space
    return XX, YY, extent, control_points, field_t, contours
