# cost_matrix.py
import numpy as np
import math

def calculate_cost_matrix(tracked_objects, detections, pred_weight, upward_tolerance):
    cost_matrix = np.zeros((len(tracked_objects), len(detections)), dtype=np.float32)
    for i, (track_id, (cx, cy, _, _, _, prev_cy, _)) in enumerate(tracked_objects.items()):
        pred_cy = cy + (cy - prev_cy)
        pred_cx = cx
        for j, (detection_cx, detection_cy, _) in enumerate(detections):
            dist_actual = math.hypot(detection_cx - cx, detection_cy - cy)
            dist_pred = math.hypot(detection_cx - pred_cx, detection_cy - pred_cy)
            distance = (1 - pred_weight) * dist_actual + pred_weight * dist_pred
            if detection_cy < prev_cy - upward_tolerance:
                distance = 9999
            cost_matrix[i, j] = distance
    return cost_matrix
