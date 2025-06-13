import pytest
import numpy as np

import config
from navigation import calibration as calib
from navigation import grid_utils as grid

@pytest.fixture(autouse=True)
def reset_state():
    calib.inv_homography_matrix = None
    grid.obstacles.clear()
    yield
    grid.obstacles.clear()

def test_pixel_to_cm_without_homography():
    assert grid.pixel_to_cm(10, 20) is None

def test_pixel_to_cm_returns_none_if_no_homography():
    # without a homography installed we should get None
    assert grid.pixel_to_cm(10, 20) is None

def test_pixel_to_cm_identity_homography():
    # stub an identity homography so pixels map 1:1 to cm
    H = np.eye(3, dtype="float32")
    calib.inv_homography_matrix = H

    # pick any pixel coords
    x, y = 123.0, 45.0
    out = grid.pixel_to_cm(x, y)
    # should round-trip exactly
    assert pytest.approx(out[0], rel=1e-6) == x
    assert pytest.approx(out[1], rel=1e-6) == y

def test_cm_to_grid_coords():
    # if GRID_SPACING_CM is 2, then 5cm → cell 2
    assert grid.cm_to_grid_coords(5, 5) == (5 // config.GRID_SPACING_CM, 5 // config.GRID_SPACING_CM)

def test_ensure_outer_edges_walkable_clears_border():
    # seed some border and interior obstacles
    max_x = config.REAL_WIDTH_CM // config.GRID_SPACING_CM
    grid.obstacles.update({(0,0), (max_x,0), (0,5), (max_x,5), (2,2)})
    grid.ensure_outer_edges_walkable()
    # border cells should be removed
    assert (0,0) not in grid.obstacles
    assert (max_x,0) not in grid.obstacles
    # interior cell remains
    assert (2,2) in grid.obstacles

def test_toggle_obstacle_at_pixel_adds_and_removes():
    # stub identity homography
    calib.inv_homography_matrix = np.eye(3, dtype="float32")
    # pixel (4,4) → cm (4,4) → grid cell (4//2,4//2) = (2,2)
    grid.toggle_obstacle_at_pixel(4, 4)
    assert (2,2) in grid.obstacles

    # toggling again should remove it
    grid.toggle_obstacle_at_pixel(4, 4)
    assert (2,2) not in grid.obstacles
