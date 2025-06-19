import time
import math
import hardware
import motion

class HeadingFilter:
    def __init__(self, alpha=0.9, vision_init=None):
        self.alpha = alpha

        # if the caller gives us an initial vision heading, use it…
        if vision_init is not None:
            hardware.gyro.reset()
            time.sleep(0.05)
            hardware.gyro_offset = vision_init
            # and force both our filter state
            self.angle     = vision_init
            self._last_raw = (-hardware.gyro.angle) % 360
        else:
            # fall back to whatever the current fused gyro says
            self.angle     = hardware.get_heading()
            self._last_raw = (-hardware.gyro.angle) % 360

        self._last_time = time.time()

    def reset(self, vision_init):
        """Hard-reset the filter to a new ArUco zero."""
        self._init_state(vision_init)
    
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
        if (motion.robot_pose["theta"] is not None and
            (now - motion.robot_pose["timestamp"]) < 0.5):
            θ_vision = motion.robot_pose["theta"]
            # complementary filter step
            self.angle = (self.alpha * pred + (1 - self.alpha) * θ_vision) % 360
        else:
            self.angle = pred

        return self.angle
