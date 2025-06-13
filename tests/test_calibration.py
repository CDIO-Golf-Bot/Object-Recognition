# tests/test_calibration.py

import os, sys
import pytest
import numpy as np
import cv2

import navigation.calibration as calibration  # your calibration.py
import config       # top-level config.py


@pytest.fixture(autouse=True)
def reset_calibration(monkeypatch):
    # Reset all globals before each test
    calibration.calibration_points.clear()
    calibration.homography_matrix = None
    calibration.inv_homography_matrix = None
    calibration.grid_overlay = None
    yield
    calibration.calibration_points.clear()
    calibration.homography_matrix = None
    calibration.inv_homography_matrix = None
    calibration.grid_overlay = None


def test_click_to_set_corners_accumulates_points_and_computes_homography(monkeypatch):
    # Stub cv2.getPerspectiveTransform → identity, np.linalg.inv → identity
    monkeypatch.setattr(cv2, "getPerspectiveTransform", lambda dst, src: np.eye(3, dtype="float32"))
    monkeypatch.setattr(np.linalg, "inv",             lambda m: np.eye(3, dtype="float32"))
    # Stub warpPerspective to return a known array
    dummy = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "warpPerspective", lambda canvas, M, size, flags=None: dummy)

    # Simulate 4 clicks
    pts = [(10, 20), (30, 20), (30, 40), (10, 40)]
    for x, y in pts:
        calibration.click_to_set_corners(cv2.EVENT_LBUTTONDOWN, x, y, None, None)

    # Should have recorded 4 points
    assert calibration.calibration_points == [list(p) for p in pts]
    # Should have set a homography and its inverse
    assert calibration.homography_matrix is not None
    assert np.allclose(calibration.inv_homography_matrix, np.eye(3, dtype="float32"))
    # And grid_overlay should be our dummy
    assert calibration.grid_overlay is dummy


def test_click_to_set_corners_right_click_removes_last_point():
    # Preseed with two points
    calibration.calibration_points[:] = [[0, 0], [1, 1]]
    # Right‐click
    calibration.click_to_set_corners(cv2.EVENT_RBUTTONDOWN, 0, 0, None, None)
    # Should remove the last one only
    assert calibration.calibration_points == [[0, 0]]


def test_get_aruco_robot_position_and_heading_no_marker():
    # A blank frame → no markers → should return None
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert calibration.get_aruco_robot_position_and_heading(frame) is None


def test_create_and_cache_grid_overlay_without_homography(monkeypatch):
    # If homography_matrix is None, warpPerspective should never be called,
    # and grid_overlay remains None
    called = False
    def fake_warp(*args, **kwargs):
        nonlocal called
        called = True
        return np.array([])
    monkeypatch.setattr(cv2, "warpPerspective", fake_warp)

    calibration.homography_matrix = None
    calibration.create_and_cache_grid_overlay()
    assert calibration.grid_overlay is None
    assert not called
