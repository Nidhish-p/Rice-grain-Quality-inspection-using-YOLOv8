# main.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment  # (import present in your original file)
import math
from collections import deque

from utils.cost_matrix import calculate_cost_matrix
from utils.update_steps import (
    retire_crossed_bottom,
    filter_detections_above_bottom,
    run_assignment,
    build_match_sets,
    update_matched_tracks,
    handle_unmatched_tracks,
    create_new_tracks,
)

# ------------ device selection (CUDA if available else CPU) ------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------ Initialize video and model ------------
video_path = 'demo/input_video.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('models/bestv8m.pt')
model.to(device)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('demo/main_outputm.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

# ------------ Tracking settings ------------
max_distance = 50
lost_frames_threshold = 7
confirmation_frames = 18
next_track_id = 0

tracked_objects = {}   # track_id -> (cx, cy, class_id, lost_frames, age, prev_cy, class_deque)
confirmed_grains = set()  # Holds (track_id, final_class)
retired_ids = set()

bottom_threshold = int(height * 0.7)
upward_tolerance = 8  # Allow small upward movement in pixels
pred_weight = 0.5     # Weight for predicted position in matching

# -------------------- MAIN LOOP --------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ------------ run YOLO inference ------------
    results = model(frame)

    # ------------ build detections list ------------
    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()

    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        label = int(labels[i]) if i < len(labels) else 0
        if cy <= bottom_threshold:
            detections.append((cx, cy, label))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ------------ retire grains crossing bottom ------------
    retire_crossed_bottom(tracked_objects, confirmed_grains, retired_ids, bottom_threshold)

    # ------------ filter detections above bottom ------------
    detections = filter_detections_above_bottom(detections, bottom_threshold)

    # ------------ Hungarian assignment ------------
    cost_matrix, row_ind, col_ind = run_assignment(
        tracked_objects, detections, calculate_cost_matrix, pred_weight, upward_tolerance
    )

    # ------------ matching detections to tracks ------------
    if cost_matrix is None:
        unmatched_tracks = set(tracked_objects.keys())
        unmatched_detections = set(range(len(detections)))
        matched = set()
    else:
        matched, unmatched_tracks, unmatched_detections = build_match_sets(
            tracked_objects, detections, row_ind, col_ind, cost_matrix, max_distance
        )

    # ------------ update matched tracks ------------
    update_matched_tracks(tracked_objects, detections, matched, confirmation_frames, confirmed_grains)

    # ------------ handle unmatched tracks ------------
    handle_unmatched_tracks(
        tracked_objects, unmatched_tracks, lost_frames_threshold, bottom_threshold, confirmed_grains, retired_ids
    )

    # ------------ create new tracks ------------
    next_track_id = create_new_tracks(
        tracked_objects, unmatched_detections, detections, retired_ids, next_track_id, confirmation_frames, max_distance
    )

    # ------------ draw info (IDs and classes for confirmed) ------------
    for track_id, (cx, cy, class_id, lost_frames, age, prev_cy, class_deque) in tracked_objects.items():
        if track_id in retired_ids:
            continue

        display_class = None
        for cid, final_class in confirmed_grains:
            if cid == track_id:
                display_class = final_class
                break

        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(frame, f'ID:{track_id}', (int(cx), int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if display_class is not None:
            cv2.putText(frame, f'Class:{display_class}', (int(cx), int(cy) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ------------ write frame ------------
    out.write(frame)

# ------------ cleanup ------------
cap.release()
out.release()

# ------------ Final grain counts ------------
class_counts = {}
for _, class_id in confirmed_grains:
    class_counts[class_id] = class_counts.get(class_id, 0) + 1

print("Grain counts by class:")
for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count}")
