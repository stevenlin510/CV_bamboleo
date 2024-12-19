import cv2
import numpy as np
from typing import List, Tuple, Dict
import argparse

class ObjectTracker:
    def __init__(self):
        self.trackers = {}
        self.tracking_objects = {}  
        self.next_id = 0

    def create_tracker(self):
        tracker = cv2.legacy.TrackerCSRT_create()
        return tracker

    def add_object(self, frame, bbox):
        tracker = self.create_tracker()
        if tracker.init(frame, bbox):
            self.trackers[self.next_id] = tracker
            self.tracking_objects[self.next_id] = bbox
            self.next_id += 1

    def update_all(self, frame):
        to_delete = []
        updated_objects = {}

        for obj_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                updated_objects[obj_id] = tuple(map(int, bbox))
            else:
                to_delete.append(obj_id)

        for obj_id in to_delete:
            del self.trackers[obj_id]
            del self.tracking_objects[obj_id]

        self.tracking_objects = updated_objects
        return updated_objects

def detect_yellow_board(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None

def merge_overlapping_boxes(boxes, overlap_threshold=0.5, size_threshold=50):
    merged_boxes = []
    while boxes:
        x1, y1, w1, h1 = boxes.pop(0)
        merged = False
        for i, (x2, y2, w2, h2) in enumerate(merged_boxes):
           
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            area1 = w1 * h1
            area2 = w2 * h2
            if overlap_area / min(area1, area2) > overlap_threshold:
                
                merged_boxes[i] = (min(x1, x2), min(y1, y2), 
                                  max(x1 + w1, x2 + w2) - min(x1, x2), 
                                  max(y1 + h1, y2 + h2) - min(y1, y2))
                merged = True
                break
        if not merged:
            merged_boxes.append((x1, y1, w1, h1))
    
    filtered_boxes = [
        (x, y, w, h) for x, y, w, h in merged_boxes 
        if w > size_threshold and h > size_threshold and 0.5 < w / h < 2.0
    ]

    return merged_boxes

def detect_objects_with_improvements(frame, board_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    combined_mask = cv2.bitwise_or(black_mask, red_mask)
    combined_mask = cv2.bitwise_and(combined_mask, board_mask)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=4)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    boxes = []
    for i in range(1, num_labels): 
        x, y, w, h, area = stats[i]
        if area > 50:
            boxes.append((x, y, w, h))
    
    boxes = merge_overlapping_boxes(boxes, overlap_threshold=0.3, size_threshold=50)

    result_frame = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return result_frame, boxes

def process_frame(frame, object_tracker, detection_interval, frame_count):
    output = frame.copy()

    board_contour = detect_yellow_board(frame)
    if board_contour is None:
        return output

    cv2.drawContours(output, [board_contour], -1, (0, 255, 255), 2)
    board_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(board_mask, [board_contour], -1, (255), -1)

    if frame_count % detection_interval == 0:
        _, new_boxes = detect_objects_with_improvements(frame, board_mask)
        for bbox in new_boxes:
            if len(bbox) != 4:
                print(f"Invalid bbox format: {bbox}")
                continue

            is_new = True
            for tracked_bbox in object_tracker.tracking_objects.values():
                x1, y1, w1, h1 = bbox
                x2, y2, w2, h2 = tracked_bbox
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                if overlap_area > 0:  
                    is_new = False
                    break
            if is_new:
                print(f"Adding bbox: {bbox}")  
                object_tracker.add_object(frame, bbox)

    tracked_objects = object_tracker.update_all(frame)
    for obj_id, bbox in tracked_objects.items():
        x, y, w, h = bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, f"ID:{obj_id}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return output

def main(args):
    source = args.video_input
    cap = cv2.VideoCapture(source)
    object_tracker = ObjectTracker()
    detection_interval = 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        output = process_frame(frame, object_tracker, detection_interval, frame_count)
        cv2.imshow('Object Tracking', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_input', type=str, required=True, help='Path to the input video directory')
    args = parser.parse_args()

    main(args)
