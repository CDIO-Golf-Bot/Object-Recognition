# tests/test_planner.py
import pytest
import itertools

import navigation.planner as planner
import navigation.grid_utils as gu
import navigation.calibration as cal
import config

@pytest.fixture(autouse=True)
def reset_planner_state(monkeypatch):
    # Reset all module‐level state in planner
    attrs = [
        "cached_route", "full_grid_path", "pending_route",
        "last_ball_positions_cm", "last_selected_goal",
        "robot_position_cm", "selected_goal"
    ]
    for a in attrs:
        setattr(planner, a, None if a.endswith("_cm") or a.endswith("_route") else [])
    # Clear obstacles
    gu.obstacles.clear()
    # Reset calibration
    cal.inv_homography_matrix = None
    cal.homography_matrix     = None
    yield
    # cleanup again
    gu.obstacles.clear()

def test_heuristic():
    # Manhattan distance
    assert planner.heuristic((0,0), (2,3)) == 5
    assert planner.heuristic((5,5), (2,1)) == 7

def test_astar_without_obstacles():
    # Simple straight‐line on a 5×5 grid from (0,0) to (2,0)
    path = planner.astar((0,0), (2,0), grid_w=5, grid_h=5, obstacles_set=set())
    # Expect cells [(1,0),(2,0)]
    assert path == [(1,0),(2,0)]

def test_astar_with_obstacle_blocking_direct():
    # Block (1,0) on the way, should route around via (0,1),(1,1),(2,1),(2,0)
    obstacles = {(1,0)}
    path = planner.astar((0,0), (2,0), grid_w=5, grid_h=5, obstacles_set=obstacles)
    # Must avoid (1,0)
    assert (1,0) not in path
    assert path[-1] == (2,0)

def test_compress_path():
    line_then_down = [(0,0),(1,0),(2,0),(2,1),(2,2)]
    compressed = planner.compress_path(line_then_down)
    # Should keep only endpoints of straight segments: [(0,0),(2,0),(2,2)]
    assert compressed == [(0,0),(2,0),(2,2)]

def test_pick_top_n(monkeypatch):
    # Place robot at (0,0)
    monkeypatch.setattr(planner, "robot_position_cm", (0,0))
    balls = [
        (1,0,"b1"),  # dist 1
        (0,2,"b2"),  # dist 2
        (3,3,"b3"),  # dist 6
    ]
    top2 = planner.pick_top_n(balls, n=2)
    # Should be the two closest
    labels = [b[2] for b in top2]
    assert labels == ["b1","b2"]
    
def test_get_expanded_obstacles():
    raw = {(1, 1)}
    out = planner.get_expanded_obstacles(raw)

    # build the expected neighborhood based on the real BUFFER_CELLS
    b = config.BUFFER_CELLS
    max_g = config.REAL_WIDTH_CM // config.GRID_SPACING_CM
    max_h = config.REAL_HEIGHT_CM // config.GRID_SPACING_CM

    expected = {
        (i, j)
        for i in range(1 - b, 1 + b + 1)
        for j in range(1 - b, 1 + b + 1)
        if 0 <= i <= max_g and 0 <= j <= max_h
    }

    assert out == expected


def test_compute_best_route_empty():
    # No balls → empty route
    route_cm, full_cells = planner.compute_best_route([], "A")
    assert route_cm == []
    assert full_cells == []

def test_compute_best_route_simple(monkeypatch):
    # Shrink the world to 2×1 grid: START=(0,0), GOAL 'A'=(2,0)
    monkeypatch.setattr(config, "START_POINT_CM", (0,0))
    monkeypatch.setattr(config, "GOAL_RANGE", {"A":[(2,0)]})
    monkeypatch.setattr(config, "REAL_WIDTH_CM", 2)
    monkeypatch.setattr(config, "REAL_HEIGHT_CM", 1)
    monkeypatch.setattr(config, "GRID_SPACING_CM", 1)

    # No obstacles
    gu.obstacles.clear()

    # One ball at (1,0)
    ball_list = [(1,0,"unused_label",0,0)]
    route_cm, full_cells = planner.compute_best_route(ball_list, "A")

    # Expect cm route [ (0,0),(1,0),(2,0) ]
    assert route_cm == [(0,0),(1,0),(2,0)]
    # Expect grid cells [(1,0),(2,0)]
    assert full_cells == [(1,0),(2,0)]

def test_save_route_to_file(tmp_path):
    # Write out to a temp file
    route = [(0,0),(1,2),(3,4)]
    fname = tmp_path / "route.txt"
    planner.save_route_to_file(route, filename=str(fname))

    text = fname.read_text().splitlines()
    # Should have three lines with formatted floats
    assert text == ["0.00,0.00","1.00,2.00","3.00,4.00"]
