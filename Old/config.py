class Config:
    # Roboflow settings
    API_KEY = "7kMjalIwU9TqGmKM0g4i"
    WORKSPACE = "pingpong-fafrv"
    PROJECT_NAME = "pingpongdetector-rqboj"
    MODEL_VERSION = 2

    # Camera settings
    CAMERA_INDEX = 1
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30
    BUFFER_SIZE = 1

    # Grid settings (cm)
    REAL_WIDTH_CM = 180
    REAL_HEIGHT_CM = 120
    GRID_SPACING_CM = 2

    # Calibration
    CALIBRATION_POINTS_REQUIRED = 4

    # Ignored area in cm
    IGNORED_AREA = {
        'x_min': 50,
        'x_max': 100,
        'y_min': 50,
        'y_max': 100
    }

    # Start point in cm
    START_POINT_CM = (20, 20)

    # Goal ranges in cm
    GOAL_RANGE = {
        'A': [],
        # 'B': [(REAL_WIDTH_CM - 5, y) for y in range(56, 65)]
    }
