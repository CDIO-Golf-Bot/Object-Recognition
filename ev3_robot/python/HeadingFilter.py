import time
import math
import hardware
from motion import robot_pose

class HeadingFilter:
    def __init__(self, alpha=0.9):
        """
        alpha: how much to trust the gyro prediction vs. the ArUco measurement.
        0.0 = pure ArUco (slow, no drift)
        1.0 = pure gyro   (fast, drifts)
        """
        self.alpha = alpha
        # Initialize both to your current best heading:
        self.angle = hardware.get_heading()  # (negated+offset gyro)
        # track the raw gyro angle to compute deltas
        self._last_raw = (-hardware.gyro.angle) % 360
        self._last_time = time.time()

    def update(self):
        now = time.time()
        dt  = now - self._last_time
        self._last_time = now

        # 1) Fast prediction: integrate gyro delta
        raw = (-hardware.gyro.angle) % 360
        # shortest‐path difference (–180…+180)
        dθ  = ((raw - self._last_raw + 180) % 360) - 180
        self._last_raw = raw

        pred = (self.angle + dθ) % 360

        # 2) Slow correction: if ArUco is fresh, blend in its heading
        if (robot_pose["theta"] is not None and
            (now - robot_pose["timestamp"]) < 0.5):
            θ_vision = robot_pose["theta"]
            # complementary filter step
            self.angle = (self.alpha * pred + (1 - self.alpha) * θ_vision) % 360
        else:
            self.angle = pred

        return self.angle
