import cv2
import numpy as np
from config import Config

class Visualizer:
    def __init__(self, config, grid_manager):
        self.config = config
        self.grid = grid_manager
        self.selected_goal = 'A'
        self.window = "Live Object Detection"

    def start(self, frame_queue, detector, pathfinder, communicator):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._mouse_callback)
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                dets = detector.detect(frame)

                out = self._draw_grid(frame)
                out = self._draw_detections(out, dets)

                cm_list=[]
                for d in dets:
                    cm = self.grid.pixel_to_cm(*d['center_px'])
                    if cm and not (
                        self.config.IGNORED_AREA['x_min']<=cm[0]<=self.config.IGNORED_AREA['x_max'] and
                        self.config.IGNORED_AREA['y_min']<=cm[1]<=self.config.IGNORED_AREA['y_max']
                    ):
                        cm_list.append((cm[0],cm[1],d['label']))

                full_path = self._compute_full_path(cm_list, pathfinder)
                out = self._draw_route(out, full_path)
                cv2.imshow(self.window, out)

            key = cv2.waitKey(1)&0xFF
            if key==ord('q'):
                break
            elif key==ord('1'):
                self.selected_goal='A'
            elif key==ord('2'):
                self.selected_goal='B'
            elif key==ord('s'):
                communicator.send(full_path, 'N')

        cv2.destroyAllWindows()

    def _mouse_callback(self, event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.grid.add_calibration_point((x,y))
        elif event==cv2.EVENT_RBUTTONDOWN and self.grid.h_matrix is not None:
            cm=self.grid.pixel_to_cm(x,y)
            if cm:
                gx,gy=self.grid.cm_to_grid(*cm)
                self.grid.toggle_obstacle(gx,gy)

    def _draw_grid(self, frame):
        if self.grid.h_matrix is None:
            return frame
        overlay=frame.copy()
        for x in range(0,self.grid.real_w+1,self.grid.spacing):
            p1=cv2.perspectiveTransform(np.array([[[x,0]]],dtype='float32'),self.grid.h_matrix)[0][0]
            p2=cv2.perspectiveTransform(np.array([[[x,self.grid.real_h]]],dtype='float32'),self.grid.h_matrix)[0][0]
            cv2.line(overlay,tuple(p1.astype(int)),tuple(p2.astype(int)),(100,100,100),1)
        for y in range(0,self.grid.real_h+1,self.grid.spacing):
            p1=cv2.perspectiveTransform(np.array([[[0,y]]],dtype='float32'),self.grid.h_matrix)[0][0]
            p2=cv2.perspectiveTransform(np.array([[[self.grid.real_w,y]]],dtype='float32'),self.grid.h_matrix)[0][0]
            cv2.line(overlay,tuple(p1.astype(int)),tuple(p2.astype(int)),(100,100,100),1)
        return overlay

    def _draw_detections(self, frame, dets):
        for d in dets:
            x1,y1,x2,y2=d['bbox']
            cv2.rectangle(frame,(x1,y1),(x2,y2),d['color'],2)
            cv2.putText(frame,d['label'],(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,d['color'],2)
        return frame

    def _compute_full_path(self, cm_list, pathfinder):
        non_orange=[b for b in cm_list if b[2].lower()!='orange']
        orange=[b for b in cm_list if b[2].lower()=='orange']
        route=[self.config.START_POINT_CM]
        curr=route[0]
        while non_orange:
            nxt=min(non_orange,key=lambda b: pathfinder.heuristic(pathfinder.grid.cm_to_grid(curr[0],curr[1]),pathfinder.grid.cm_to_grid(b[0],b[1])))
            route.append((nxt[0],nxt[1])); non_orange.remove(nxt); curr=(nxt[0],nxt[1])
        if orange:
            route.append((orange[0][0],orange[0][1])); curr=(orange[0][0],orange[0][1])
        goal_pts=self.config.GOAL_RANGE[self.selected_goal]
        goal=min(goal_pts,key=lambda g: pathfinder.heuristic(pathfinder.grid.cm_to_grid(curr[0],curr[1]),pathfinder.grid.cm_to_grid(g[0],g[1])))
        route.append(goal)
        full_path=[]
        for i in range(len(route)-1):
            s=pathfinder.grid.cm_to_grid(route[i][0],route[i][1])
            e=pathfinder.grid.cm_to_grid(route[i+1][0],route[i+1][1])
            seg=pathfinder.astar(s,e)
            if full_path and seg and seg[0]==full_path[-1]: seg=seg[1:]
            full_path.extend(seg)
        return full_path

    def _draw_route(self, frame, path):
        if self.grid.h_matrix is None or not path:
            return frame
        overlay=frame.copy(); total=0
        prev=None
        for cell in path:
            pt_cm=np.array([[[cell[0]*self.grid.spacing,cell[1]*self.grid.spacing]]],dtype='float32')
            pt_px=cv2.perspectiveTransform(pt_cm,self.grid.h_matrix)[0][0]; pt=tuple(pt_px.astype(int))
            if prev: cv2.line(overlay,prev,pt,(0,255,255),2); total+=self.grid.spacing
            prev=pt
        cv2.putText(overlay,f"Total: {total}cm Goal {self.selected_goal}",(10,overlay.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        return overlay