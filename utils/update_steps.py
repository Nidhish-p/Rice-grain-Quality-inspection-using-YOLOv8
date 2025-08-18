# update_steps.py
from collections import deque
from scipy.optimize import linear_sum_assignment
import math

# ------------ retire grains crossing bottom ------------
def retire_crossed_bottom(tracked_objects, confirmed_grains, retired_ids, bottom_threshold):
    for tid in list(tracked_objects.keys()):
        cx, cy, class_id, lost_frames, age, prev_cy, class_deque = tracked_objects[tid]
        if cy > bottom_threshold:
            if tid not in [g[0] for g in confirmed_grains]:
                # Not confirmed yet → use mode of deque so far
                final_class = max(set(class_deque), key=class_deque.count) if len(class_deque) > 0 else class_id
                confirmed_grains.add((tid, final_class))
            retired_ids.add(tid)
            del tracked_objects[tid]

# ------------ filter detections above bottom ------------
def filter_detections_above_bottom(detections, bottom_threshold):
    return [det for det in detections if det[1] < bottom_threshold]

# ------------ Hungarian assignment (build cost + solve) ------------
def run_assignment(tracked_objects, detections, calculate_cost_matrix_fn, pred_weight, upward_tolerance):
    if tracked_objects and detections:
        cost_matrix = calculate_cost_matrix_fn(tracked_objects, detections, pred_weight, upward_tolerance)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        cost_matrix = None
        row_ind, col_ind = [], []
    return cost_matrix, row_ind, col_ind

# ------------ build match sets (threshold by max_distance) ------------
def build_match_sets(tracked_objects, detections, row_ind, col_ind, cost_matrix, max_distance):
    unmatched_tracks = set(tracked_objects.keys())
    unmatched_detections = set(range(len(detections)))
    matched = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_distance:
            matched.add((r, c))
            unmatched_tracks.remove(list(tracked_objects.keys())[r])
            unmatched_detections.remove(c)

    return matched, unmatched_tracks, unmatched_detections

# ------------ update matched tracks (age, positions, class deque, confirmations) ------------
def update_matched_tracks(tracked_objects, detections, matched, confirmation_frames, confirmed_grains):
    for r, c in matched:
        track_id = list(tracked_objects.keys())[r]
        detection_cx, detection_cy, class_id = detections[c]

        prev_cy = tracked_objects[track_id][1]  # old cy
        class_deque = tracked_objects[track_id][6]
        class_deque.append(class_id)

        tracked_objects[track_id] = (
            detection_cx, detection_cy, class_id,
            0,
            tracked_objects[track_id][4] + 1,
            detection_cy,
            class_deque
        )

        # Confirm after confirmation_frames
        if tracked_objects[track_id][4] == confirmation_frames:
            final_class = max(set(class_deque), key=class_deque.count)
            confirmed_grains.add((track_id, final_class))

# ------------ handle unmatched tracks (lost frames & retire if bottom) ------------
def handle_unmatched_tracks(tracked_objects, unmatched_tracks, lost_frames_threshold, bottom_threshold, confirmed_grains, retired_ids):
    for track_id in list(unmatched_tracks):
        cx, cy, class_id, lost_frames, age, prev_cy, class_deque = tracked_objects[track_id]
        lost_frames += 1
        tracked_objects[track_id] = (cx, cy, class_id, lost_frames, age, prev_cy, class_deque)

        if cy > bottom_threshold:
            if track_id not in [g[0] for g in confirmed_grains]:
                final_class = max(set(class_deque), key=class_deque.count) if len(class_deque) > 0 else class_id
                confirmed_grains.add((track_id, final_class))
            retired_ids.add(track_id)
            del tracked_objects[track_id]
        elif lost_frames >= lost_frames_threshold:
            del tracked_objects[track_id]

# ------------ create new tracks for unmatched detections ------------
def create_new_tracks(tracked_objects, unmatched_detections, detections, retired_ids, next_track_id, confirmation_frames, max_distance):
    for i in unmatched_detections:
        detection_cx, detection_cy, class_id = detections[i]
        is_new_grain = True

        for tid, (cx, cy, _, _, _, _, _) in tracked_objects.items():
            if tid in retired_ids:
                continue
            if math.hypot(detection_cx - cx, detection_cy - cy) < max_distance:
                is_new_grain = False
                break

        if is_new_grain:
            tracked_objects[next_track_id] = (
                detection_cx,
                detection_cy,
                class_id,
                0,
                1,
                detection_cy,
                deque([class_id], maxlen=confirmation_frames)
            )
            next_track_id += 1

    return next_track_id
