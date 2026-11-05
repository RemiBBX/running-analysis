import time

import torch

from running_analysis.visualization import draw_keypoints, draw_skeleton


def process_frame(frame, detector, pose_estimator):
    torch.mps.synchronize()
    start_time1 = time.perf_counter()
    person_boxes = detector.predict(frame)
    torch.mps.synchronize()
    end_time1 = time.perf_counter()
    detection_time = end_time1 - start_time1

    n = len(person_boxes)
    print(f"Nombre de personnes détectées : {n}")
    if n == 0:
        print(f"Detection       : {detection_time * 1000:.2f} ms")
        return frame

    torch.mps.synchronize()
    start_time2 = time.perf_counter()
    pose_results = pose_estimator.predict(frame, person_boxes)
    torch.mps.synchronize()
    end_time2 = time.perf_counter()
    estimation_time = end_time2 - start_time2

    torch.mps.synchronize()
    start_time3 = time.perf_counter()
    for i, person in enumerate(pose_results):
        frame = draw_skeleton(frame, person)
        frame = draw_keypoints(frame, person)

    torch.mps.synchronize()
    end_time3 = time.perf_counter()
    visualization_time = end_time3 - start_time3

    total_time = detection_time + estimation_time + visualization_time
    processing_fps = 1 / total_time if total_time > 0 else 0

    print(f"Detection       : {detection_time * 1000:.2f} ms")
    print(f"Pose estimation : {estimation_time * 1000:.2f} ms")
    print(f"Visualization   : {visualization_time * 1000:.2f} ms")
    print(
        f"Total           : {(detection_time + estimation_time + visualization_time) * 1000:.2f} ms"
    )
    print(f"Processing FPS  : {processing_fps:.2f}")
    print("-" * 40)

    return frame
