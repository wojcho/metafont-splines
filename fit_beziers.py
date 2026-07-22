import numpy as np
from scipy.optimize import least_squares

def bezier_cubic(
    P0: np.ndarray[np.float64],
    P1: np.ndarray[np.float64],
    P2: np.ndarray[np.float64],
    P3: np.ndarray[np.float64],
    t: np.ndarray[np.float64] | Sequence[float],
) -> np.ndarray[np.float64]:
    """
    Bezier evaluation
    """
    t = np.asarray(t)
    u = 1 - t
    return (
        (u**3)[:, None] * P0 +
        (3*u*u*t)[:, None] * P1 +
        (3*u*t*t)[:, None] * P2 +
        (t**3)[:, None] * P3
    )

def chord_length_param(points: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Stable parameterization
    """
    d = np.linalg.norm(points[1:] - points[:-1], axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] == 0:
        return np.linspace(0, 1, len(points))
    return s / s[-1]

def fit_cubic(points: np.ndarray[np.float64], t: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Initial cubic fit with linear least squares
    """
    P0, P3 = points[0], points[-1]

    u = 1 - t
    B1 = 3 * u*u*t
    B2 = 3 * u*t*t

    A = np.zeros((2*len(t), 4))
    A[0::2, 0] = B1
    A[1::2, 1] = B1
    A[0::2, 2] = B2
    A[1::2, 3] = B2

    rhs = (points - (u**3)[:, None]*P0 - (t**3)[:, None]*P3).reshape(-1)

    x, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    P1 = np.array([x[0], x[1]])
    P2 = np.array([x[2], x[3]])

    return np.array([P0, P1, P2, P3])

def max_error(
    points: np.ndarray[np.float64],
    curve: np.ndarray[np.float64],
    t: np.ndarray[np.float64],
) -> tuple[float, int]:
    """
    Error computation
    """
    C = bezier_cubic(curve[0], curve[1], curve[2], curve[3], t)
    d2 = np.sum((C - points)**2, axis=1)
    idx = np.argmax(d2)
    return d2[idx], idx

def fit_curve(
    points: np.ndarray[np.float64],
    error_tol: float=1e-3,
    depth: int=0,
    max_depth: int=20,
) -> list[np.ndarray[np.float64]]:
    """
    Recursive Schneider fitter
    """

    t = chord_length_param(points)
    curve = fit_cubic(points, t)

    err, idx = max_error(points, curve, t)

    if err < error_tol or len(points) < 4 or depth >= max_depth:
        return [curve]

    # split at worst point
    left = points[:idx+1]
    right = points[idx:]

    # avoid degenerate splits
    if len(left) < 2 or len(right) < 2:
        return [curve]

    return (
        fit_curve(left, error_tol, depth+1, max_depth) +
        fit_curve(right, error_tol, depth+1, max_depth)
    )

def beziers_to_svg_path(
    beziers: Sequence[np.ndarray[np.float64]],
    precision: int=3
) -> str:
    """
    Convert list of cubic Bezier control point arrays to an SVG path string.
    beziers: list of 4x2 arrays or sequences: [ [P0,P1,P2,P3], ... ]
    precision: decimal places for floats (int)
    Returns: SVG path string (str)
    """
    fmt = ("{:.%df}" % precision).format
    if not beziers:
        return ""
    parts = []
    # Move to first segment start
    # WARNING there is scaling of everyting by 100
    P0 = beziers[0][0] * 100
    parts.append(f"M {fmt(P0[0])} {fmt(P0[1])}")
    for bez in beziers:
        P1, P2, P3 = bez[1] * 100, bez[2] * 100, bez[3] * 100
        parts.append(f"C {fmt(P1[0])} {fmt(P1[1])}, {fmt(P2[0])} {fmt(P2[1])}, {fmt(P3[0])} {fmt(P3[1])}")
    return " ".join(parts)
