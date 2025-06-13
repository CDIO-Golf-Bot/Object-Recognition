import random
import cv2
from roboflow import Roboflow
import threading

class RoboFlowDetector:
    def __init__(self, config):
        self.rf = Roboflow(api_key=config.API_KEY)
        self.project = self.rf.workspace(config.WORKSPACE).project(config.PROJECT_NAME)
        self.model = self.project.version(config.MODEL_VERSION).model
        self.class_colors = {}
        self.config = config

    def detect(self, frame):
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (416, 416))
        preds = self.model.predict(resized, confidence=30, overlap=20).json()
        
        detections = []
        scale_x, scale_y = w / 416, h / 416
        for p in preds.get('predictions', []):
            x, y = int(p['x'] * scale_x), int(p['y'] * scale_y)
            bw, bh = int(p['width'] * scale_x), int(p['height'] * scale_y)
            label = p['class']
            color = self.class_colors.setdefault(
                label,
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )
            detections.append({
                'bbox': (x - bw // 2, y - bh // 2, x + bw // 2, y + bh // 2),
                'center_px': (x, y),
                'label': label,
                'color': color
            })
        return detections
    def process_frames(self, frame_queue, grid_manager, pathfinder, visualizer, output_queue):
        """
        - Pulls raw frames from frame_queue.
        - Runs detect().
        - Converts detections into cm coords, filters ignored areas, populates grid_manager.ball_positions.
        - Computes path via pathfinder.
        - Renders both grid & path via visualizer.
        - Pushes annotated frames into output_queue.
        """
        while True:
            frame = frame_queue.get()  # blocking
            original = frame.copy()

            # 1) detect
            detections = self.detect(frame)

            # 2) map to cm and filter
            grid_manager.ball_positions = []
            for det in detections:
                px, py = det['center_px']
                cm = grid_manager.pixel_to_cm(px, py)
                if cm is None:
                    continue
                cx, cy = cm

                # skip ignored areas:
                ia = grid_manager.ignored_area
                if ia['x_min'] <= cx <= ia['x_max'] and ia['y_min'] <= cy <= ia['y_max']:
                    continue

                grid_manager.ball_positions.append((cx, cy, det['label']))

                # draw bbox & label
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(original, (x1, y1), (x2, y2), det['color'], 2)
                cv2.putText(original, det['label'], (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, det['color'], 2)

            # 3) compute full grid path
            pathfinder.compute_full_path(grid_manager)  

            # 4) draw grid + route
            grid_img = visualizer.draw_metric_grid(original)
            out_img  = visualizer.draw_full_route(grid_img, grid_manager)

            # 5) enqueue for display
            output_queue.put(out_img)

    def start(self, frame_queue, grid_manager, pathfinder, visualizer, output_queue):
        """Launch the above loop in its own daemon thread."""
        t = threading.Thread(
            target=self.process_frames,
            args=(frame_queue, grid_manager, pathfinder, visualizer, output_queue),
            daemon=True
        )
        t.start()