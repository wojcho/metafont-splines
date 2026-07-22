from dataclasses import dataclass
import numpy as np

from metafont_df_ellipse import make_choices, find_df_and_contours
from fit_beziers import fit_curve, beziers_to_svg_path

@dataclass
class MetafontOutlineCenterline:
    curve_path: str # One centerline SVG path d
    outline_paths: list[str] # First SVG path d is masked by union of all later ones
    control_points: list[tuple[float, float]] # Control points of centerline Bezier curve
    contours: list[np.ndarray] # Contours used for obtaining of outline paths

def from_knots_to_svg_one_curve(
    knots: list[tuple[float, float]],
    *,
    ellipse_angle_deg: float = 45,
    ellipse_semi_major: float = 0.15,
    ellipse_semi_minor: float = 0.05,
    outline_fit_error: float = 1e-3,
    samples_per_segment: int = 400,
    grid_x_samples: int = 300,
    grid_y_samples: int = 200,
    grid_padding: float = 1.0,
    svg_precision: int = 2,
) -> MetafontOutlineCenterline:
    """
    Returns
    curve_path:
        SVG path of the Hobby spline.
    outline_paths:
        One SVG path for each extracted contour.
    control_points:
        Hobby spline control points.
    contours:
        Raw contour point arrays.
    """
    # Hobby spline
    control_points = make_choices(knots)
    curve_beziers = []
    for i in range(len(knots) - 1):
        curve_beziers.append(
            np.array([
                knots[i],
                control_points[2 * i],
                control_points[2 * i + 1],
                knots[i + 1],
            ])
        )
    curve_path = beziers_to_svg_path(
        curve_beziers,
        precision=svg_precision,
    )

    # Outline
    (
        XX,
        YY,
        extent,
        _,
        field,
        contours,
    ) = find_df_and_contours(
        knots,
        ellipse_angle_deg=ellipse_angle_deg,
        ellipse_semi_major=ellipse_semi_major,
        ellipse_semi_minor=ellipse_semi_minor,
        grid_x_samples=grid_x_samples,
        grid_y_samples=grid_y_samples,
        grid_padding=grid_padding,
    )

    outline_paths = []
    for contour in contours:
        beziers = fit_curve(
            contour,
            error_tol=outline_fit_error,
        )
        path = beziers_to_svg_path(
            beziers,
            precision=svg_precision,
        )
        path += " Z"
        outline_paths.append(path)

    return MetafontOutlineCenterline(
        curve_path=curve_path,
        outline_paths=outline_paths,
        control_points=control_points,
        contours=contours,
    )

if __name__ == "__main__":
    knots = [
        (0, 0),
        (1, 0),
        (0, 1),
    ]
    outline_centerline: MetafontOutlineCenterline = from_knots_to_svg_one_curve(knots)
    print(outline_centerline.curve_path)
    print(outline_centerline.outline_paths)
